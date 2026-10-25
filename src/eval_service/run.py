"""One evaluation: the teacher answers every drawn prompt, the king and the challenger read the same
tokens, and both are measured against the teacher on the answer tokens; then the win rule."""

from __future__ import annotations

import json
import math
import queue
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger as log

from common.config import settings
from eval_service import config, servers, timing
from eval_service.dataset import Sample

if TYPE_CHECKING:
    from eval_service.metrics import Sums


@dataclass(frozen=True)
class Distance:
    """How far a model is from the teacher per answer token: mean KL(teacher ‖ model) at the output,
    and the mean relative hidden-state distance |h - h_teacher| / |h_teacher| over config.LAYERS."""

    kl: float
    dist: float


@dataclass(frozen=True)
class Duel:
    challenger: Distance
    king: Distance
    noise: Distance
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def won(self) -> bool:
        return won(self.challenger, self.king)


def needed(king: Distance) -> Distance:
    """How much closer to the teacher than this king a challenger must be on each measure."""
    return Distance(kl=config.KL_MARGIN * king.kl, dist=config.DIST_MARGIN * king.dist)


def won(challenger: Distance, king: Distance) -> bool:
    """The challenger must be closer to the teacher on both measures, each by its margin. A model
    whose score is not finite diverged (inf, NaN, or it left the teacher's answer): a diverged
    challenger loses, and a diverged king loses to any challenger that did not. NaN compares
    False, so without this a king that diverges on some inputs could never be beaten."""
    if not _finite(challenger):
        return False
    if not _finite(king):
        return True
    bar = needed(king)
    return king.kl - challenger.kl >= bar.kl and king.dist - challenger.dist >= bar.dist


def _finite(d: Distance) -> bool:
    return math.isfinite(d.kl) and math.isfinite(d.dist)


DIVERGED = Distance(kl=math.inf, dist=math.inf)


class LeftAnswer(ValueError):
    """A model read the teacher's answer and did not reproduce it: the model's own doing."""

    def __init__(self, role: str, message: str) -> None:
        self.role = role
        super().__init__(message)


Scorer = Callable[..., Duel]  # (teacher, challenger, king, samples, *, on_progress=None, ...)
Progress = Callable[[int], None]
Result = tuple[dict[str, "Sums"], dict[str, Any]]
Prefetched = dict[int, tuple[list[int], str]]  # sample index -> (answer, the teacher's states file)


def duel(
    teacher_dir: Path,
    challenger_dir: Path,
    king_dir: Path,
    samples: Sequence[Sample],
    *,
    on_progress: Progress | None = None,
    prefetched: Prefetched | None = None,
    on_teacher_drained: Callable[[], None] | None = None,
) -> Duel:
    """`prefetched` answers (a Prefetch's) skip the teacher; `on_teacher_drained` is called once
    every sample has been admitted to the teacher, when its servers start to empty out."""
    models = {"teacher": teacher_dir, "king": king_dir, "challenger": challenger_dir}
    evaluation = _Evaluation(models, len(samples), on_progress, prefetched, on_teacher_drained)
    results = evaluation.run(samples)
    log.info(
        "eval timing, {} samples scored in {:.0f}s: {}",
        len(results),
        timing.stages.elapsed(),
        timing.stages.line(),
    )
    log.info("eval slowest samples: {}", timing.stages.slowest(5))
    return Duel(
        challenger=evaluation.distance(results, "challenger"),
        king=evaluation.distance(results, "king"),
        noise=evaluation.distance(results, "teacher_rerun"),
        samples=[record for _, record in results],
    )


class _Evaluation:
    """The servers, tokenizer and comparison GPUs of one duel, and the steps each sample goes
    through: the teacher's answer, the forced reads of it, and the comparison."""

    # metrics needs torch, which only the GPU box installs, so it is imported where it is used

    def __init__(
        self,
        models: dict[str, Path],
        total: int,
        on_progress: Progress | None = None,
        prefetched: Prefetched | None = None,
        on_teacher_drained: Callable[[], None] | None = None,
    ) -> None:
        from eval_service import metrics

        started = time.monotonic()
        self.groups = servers.serve(models)
        serving = time.monotonic()
        self.tokenizer = load_tokenizer(models["teacher"])
        self.eos = _eos(models["teacher"])
        # both readers' GPUs: a gguf reader leaves most of its GPU free, and the comparison is a
        # quarter of an eval's GPU time, too much for one GPU alone
        readers = (*settings().gpus("king"), *settings().gpus("challenger"))
        devices = [f"cuda:{gpu}" for gpu in readers]
        self.heads = metrics.heads(models, devices)
        log.info(
            "eval setup: servers ready in {:.0f}s, tokenizer and final heads loaded in {:.0f}s",
            serving - started,
            time.monotonic() - serving,
        )
        self.free_devices: queue.Queue[str] = queue.Queue()
        for dev in devices:
            self.free_devices.put(dev)
        # Each thread that touches a GPU gets its own cuBLAS handle and workspace there; one
        # thread per GPU keeps that to one, where 128 workers' worth of them once filled the
        # comparison GPUs.
        self.compare_on = {dev: ThreadPoolExecutor(1) for dev in devices}
        self.preemptions_at_start = self._preemptions()
        self.total, self.done, self.started, self.lock = total, 0, 0, threading.Lock()
        self.on_progress = on_progress
        self.prefetched = prefetched if prefetched is not None else {}
        self.on_teacher_drained = on_teacher_drained
        self.diverged: set[str] = set()

    def run(self, samples: Sequence[Sample]) -> list[Result]:
        """The results of the samples that went through. A sample that fails is logged and left
        out; more than MAX_FAILED_SAMPLES of them, or none scored at all, ends the eval. A king
        or challenger that left the teacher's answer is marked diverged (self.diverged): the
        sample is left out for every role, so a model could otherwise shed the prompts it does
        worst on, and the king's with them."""
        timing.stages.reset()
        pool = ThreadPoolExecutor(config.WORKERS)
        results: list[Result] = []
        failed = 0
        stop = threading.Event()
        try:
            threading.Thread(target=_report, args=(stop, len(samples)), daemon=True).start()
            if self.on_teacher_drained:
                threading.Thread(target=self._watch_drain, args=(stop,), daemon=True).start()
            # longest prompts first, so they are not the ones still running at the end; each keeps
            # its index (its seed, and whether the teacher rereads it), and results stay in order
            by_size = sorted(range(len(samples)), key=lambda i: -_size(samples[i]))
            started = {i: pool.submit(self._sample, i, samples[i]) for i in by_size}
            futures = [started[i] for i in range(len(samples))]
            for sample, future in zip(samples, futures, strict=True):
                try:
                    results.append(future.result())
                except servers.ServerExited:
                    raise  # a dead model server ends the eval; main.py blames its GPU's model
                except Exception as exc:
                    failed += 1
                    if isinstance(exc, LeftAnswer) and exc.role in ("king", "challenger"):
                        self.diverged.add(exc.role)
                    log.warning("sample {} failed ({} so far): {}", sample.id, failed, exc)
                    if failed > config.MAX_FAILED_SAMPLES:
                        raise RuntimeError(f"{failed} samples failed, the last: {exc}") from exc
        finally:
            stop.set()
            pool.shutdown(cancel_futures=True)
            for executor in self.compare_on.values():
                executor.shutdown()
        if self._preemptions() != self.preemptions_at_start:
            raise RuntimeError(
                "vLLM preempted requests: their answer rows came from a prefill pass"
            )
        if not results:
            raise RuntimeError("no sample was scored")
        return results

    def distance(self, results: list[Result], role: str) -> Distance:
        """Token-weighted means over all samples the role read; DIVERGED for a role that left
        the teacher's answer on any sample."""
        if role in self.diverged:
            return DIVERGED
        sums = [by_role[role] for by_role, _ in results if role in by_role]
        return Distance(
            kl=sum(s.kl for s in sums) / sum(s.predictions for s in sums),
            dist=sum(s.layer_mean_dist for s in sums) / max(sum(s.activations for s in sums), 1),
        )

    def _sample(self, i: int, sample: Sample) -> Result:
        paths: dict[str, str] = {}  # role -> the hidden-state file its server wrote
        started = time.monotonic()
        with self.lock:
            self.started += 1
        try:
            with timing.stages.timed("shm wait"):
                _wait_for_room()
            prompt = _render(self.tokenizer, sample.messages, sample.tools)
            if (ahead := self.prefetched.pop(i, None)) is not None:
                answer, paths["teacher"] = ahead
            else:
                with timing.stages.timed("teacher"):
                    answer, paths["teacher"] = generate(
                        self.groups["teacher"], prompt, i, sample.id
                    )
            with timing.stages.timed("read"):
                self._read_along(i, sample, prompt, answer, paths)
            with timing.stages.timed("compare"):
                sums = self._compare(prompt, answer, paths)
            timing.stages.sample(sample.id, time.monotonic() - started, len(prompt), len(answer))
            return sums, self._record(sample, prompt, answer, sums)
        finally:
            self._count()  # a failed sample counts too, so progress reaches the total
            for path in paths.values():
                servers.discard(path)

    def _watch_drain(self, stop: threading.Event) -> None:
        """Calls on_teacher_drained once every sample has started and none waits for the teacher
        any more: from here the teacher's servers only empty out, and the next eval's answers
        can fill them (main.py claims it ahead)."""
        drained = self.on_teacher_drained
        if drained is None:
            return
        line = getattr(getattr(self.groups["teacher"], "admission", None), "line", None)
        while not stop.wait(5):
            with self.lock:
                started = self.started
            if started == self.total and not line:
                try:
                    drained()
                except Exception:
                    log.exception("claiming the next eval ahead failed")
                return

    def _read_along(
        self, i: int, sample: Sample, prompt: list[int], answer: list[int], paths: dict[str, str]
    ) -> None:
        """The king, the challenger and on every RERUN_EVERY-th sample the teacher again read the
        answer, forced; each must reproduce it token for token."""
        readers = {"king": self.groups["king"], "challenger": self.groups["challenger"]}
        if i % config.RERUN_EVERY == 0:
            readers["teacher_rerun"] = self.groups["teacher"]
        with ThreadPoolExecutor(len(readers)) as pool:
            futures = {
                role: pool.submit(group.complete, prompt, len(answer), i, prompt + answer)
                for role, group in readers.items()
            }
        reads, errors = {}, []
        # every finished request's file must be discarded, even when another one failed
        for role, future in futures.items():
            try:
                reads[role], paths[role] = future.result()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        for role, read in reads.items():
            if read != answer:
                raise LeftAnswer(role, f"{sample.id}: the {role} left the teacher's answer")

    def _compare(
        self, prompt: list[int], answer: list[int], paths: dict[str, str]
    ) -> dict[str, Sums]:
        from eval_service import metrics

        with timing.stages.timed("compare wait"):
            dev = self.free_devices.get()
        try:
            # the last answer token is never fed back in, so no server wrote a row for it
            with timing.stages.timed("compare load"):
                states = {
                    role: metrics.load_states(p, prompt + answer[:-1], len(prompt) - 1)
                    for role, p in paths.items()
                }
            with timing.stages.timed("compare math"):
                return (
                    self.compare_on[dev]
                    .submit(metrics.compare, states, self.heads[dev], answer, dev)
                    .result()
                )
        finally:
            self.free_devices.put(dev)

    def _record(
        self, sample: Sample, prompt: list[int], answer: list[int], sums: dict[str, Sums]
    ) -> dict[str, Any]:
        return {
            "sample_id": sample.id,
            "prompt_tokens": len(prompt),
            "answer_tokens": len(answer),
            "ended_with_eos": answer[-1] in self.eos,
            "runs": {role: s.summary() for role, s in sums.items()},
        }

    def _count(self) -> None:
        with self.lock:
            self.done += 1
            done = self.done
            if done % 100 == 0:
                log.info("eval progress: {} of {} samples", done, self.total)
        chunks = config.PROGRESS_CHUNKS
        if self.on_progress and done * chunks // self.total > (done - 1) * chunks // self.total:
            try:
                self.on_progress(done)
            except Exception:
                log.exception("eval progress report failed")

    def _preemptions(self) -> float:
        return sum(group.preemptions() for group in self.groups.values())


def generate(
    teacher: servers.Groups, prompt: list[int], seed: int, label: str
) -> tuple[list[int], str]:
    """The teacher's answer to `prompt`, at most MAX_TOKENS tokens, and the states file it wrote."""
    room = config.MAX_MODEL_LEN - len(prompt)
    if room < 1:
        raise ValueError(f"{label}: a {len(prompt)}-token prompt leaves no room")
    return teacher.complete(prompt, min(config.MAX_TOKENS, room), seed)


class Prefetch:
    """The teacher's answers to the first PREFETCH_SAMPLES samples of an eval that has not started
    (longest prompts first, the duel's order), generated while the running eval drains its
    teacher. `result` is the `prefetched` of that eval's duel."""

    def __init__(self, teacher_dir: Path, samples: Sequence[Sample]) -> None:
        self.teacher = servers.serve({"teacher": teacher_dir})["teacher"]  # the running one
        self.tokenizer, self.samples = load_tokenizer(teacher_dir), samples
        self.pool = ThreadPoolExecutor(config.WORKERS)
        ahead = sorted(range(len(samples)), key=lambda i: -_size(samples[i]))
        self.futures = {i: self.pool.submit(self._one, i) for i in ahead[: config.PREFETCH_SAMPLES]}

    def _one(self, i: int) -> tuple[list[int], str]:
        _wait_for_room()
        prompt = _render(self.tokenizer, self.samples[i].messages, self.samples[i].tools)
        return generate(self.teacher, prompt, i, self.samples[i].id)

    def result(self) -> Prefetched:
        """Waits for every answer; a sample whose answer failed is left to the duel."""
        out: Prefetched = {}
        for i, future in self.futures.items():
            try:
                out[i] = future.result()
            except Exception as exc:
                log.warning("answering sample {} ahead failed: {}", self.samples[i].id, exc)
        self.pool.shutdown()
        return out


def load_tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer  # the GPU box's, like torch

    return AutoTokenizer.from_pretrained(model_dir)


def _render(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> list[int]:
    """Prompt token ids with the teacher's chat template and its default thinking mode."""
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, add_special_tokens=False)


def fits(tokenizer: Any) -> Callable[[Sample], bool]:
    """Whether a sample's rendered prompt leaves the whole answer budget, for the draw. A byte-level
    token covers at least one UTF-8 byte, so a prompt that short in bytes fits without rendering."""
    limit = config.MAX_MODEL_LEN - config.MAX_TOKENS
    frame = len(_render(tokenizer, [{"role": "user", "content": ""}]))

    def check(sample: Sample) -> bool:
        size = sum(len(m["content"].encode()) for m in sample.messages)
        # tool schemas and tool calls render outside the contents, so an agent sample is rendered
        return (not sample.tools and size + frame * len(sample.messages) <= limit) or (
            len(_render(tokenizer, sample.messages, sample.tools)) <= limit
        )

    return check


def _eos(model_dir: Path) -> set[int]:
    eos = json.loads((model_dir / "generation_config.json").read_text())["eos_token_id"]
    return set(eos if isinstance(eos, list) else [eos])


def _size(sample: Sample) -> int:
    return sum(len(m["content"]) for m in sample.messages)


def _report(stop: threading.Event, total: int) -> None:
    """Every TIMING_REPORT_S until `stop`: the running eval's per-stage timings."""
    while not stop.wait(config.TIMING_REPORT_S):
        try:
            log.info(
                "eval timing, {} of {} samples done: {}",
                timing.stages.done(),
                total,
                timing.stages.line(),
            )
        except Exception:
            log.exception("eval timing report failed")


def _wait_for_room() -> None:
    # raw states wait in /dev/shm until compared; hold new samples back when it fills up
    while True:
        shm = shutil.disk_usage("/dev/shm")
        if shm.used < 0.6 * shm.total:
            return
        time.sleep(5)
