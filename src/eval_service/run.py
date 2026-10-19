"""One evaluation: the teacher answers every drawn prompt, the king and the challenger read the same
tokens, and both are measured against the teacher on the answer tokens; then the win rule."""

from __future__ import annotations

import json
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


def won(challenger: Distance, king: Distance) -> bool:
    """The challenger must be closer to the teacher on both measures, each by its own margin."""
    return (
        king.kl - challenger.kl >= config.KL_MARGIN
        and king.dist - challenger.dist >= config.DIST_MARGIN
    )


Scorer = Callable[..., Duel]  # (teacher, challenger, king, samples, *, on_progress=None)
Progress = Callable[[int], None]
Result = tuple[dict[str, "Sums"], dict[str, Any]]


def duel(
    teacher_dir: Path,
    challenger_dir: Path,
    king_dir: Path,
    samples: Sequence[Sample],
    *,
    on_progress: Progress | None = None,
) -> Duel:
    models = {"teacher": teacher_dir, "king": king_dir, "challenger": challenger_dir}
    evaluation = _Evaluation(models, len(samples), on_progress)
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
        self, models: dict[str, Path], total: int, on_progress: Progress | None = None
    ) -> None:
        from eval_service import metrics

        started = time.monotonic()
        self.groups = servers.serve(models)
        serving = time.monotonic()
        self.tokenizer = load_tokenizer(models["teacher"])
        self.eos = _eos(models["teacher"])
        devices = [f"cuda:{gpu}" for gpu in settings().gpus("challenger")]
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
        self.total, self.done, self.lock = total, 0, threading.Lock()
        self.on_progress = on_progress

    def run(self, samples: Sequence[Sample]) -> list[Result]:
        """The results of the samples that went through. A sample that fails is logged and left
        out; more than MAX_FAILED_SAMPLES of them, or none scored at all, ends the eval."""
        timing.stages.reset()
        pool = ThreadPoolExecutor(config.WORKERS)
        results: list[Result] = []
        failed = 0
        stop = threading.Event()
        try:
            threading.Thread(target=_report, args=(stop, len(samples)), daemon=True).start()
            # longest prompts first, so they are not the ones still running at the end; each keeps
            # its index (its seed, and whether the teacher rereads it), and results stay in order
            by_size = sorted(range(len(samples)), key=lambda i: -_size(samples[i]))
            started = {i: pool.submit(self._sample, i, samples[i]) for i in by_size}
            futures = [started[i] for i in range(len(samples))]
            for sample, future in zip(samples, futures, strict=True):
                try:
                    results.append(future.result())
                except Exception as exc:
                    failed += 1
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
        """Token-weighted means over all samples the role read."""
        sums = [by_role[role] for by_role, _ in results if role in by_role]
        return Distance(
            kl=sum(s.kl for s in sums) / sum(s.predictions for s in sums),
            dist=sum(s.layer_mean_dist for s in sums) / max(sum(s.activations for s in sums), 1),
        )

    def _sample(self, i: int, sample: Sample) -> Result:
        paths: dict[str, str] = {}  # role -> the hidden-state file its server wrote
        started = time.monotonic()
        try:
            with timing.stages.timed("shm wait"):
                _wait_for_room()
            prompt = _render(self.tokenizer, sample.messages)
            with timing.stages.timed("teacher"):
                answer = self._answer(i, sample, prompt, paths)
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

    def _answer(
        self, i: int, sample: Sample, prompt: list[int], paths: dict[str, str]
    ) -> list[int]:
        room = config.MAX_MODEL_LEN - len(prompt)
        if room < 1:
            raise ValueError(f"{sample.id}: a {len(prompt)}-token prompt leaves no room")
        answer, paths["teacher"] = self.groups["teacher"].complete(
            prompt, min(config.MAX_TOKENS, room), i
        )
        return answer

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
                raise ValueError(f"{sample.id}: the {role} left the teacher's answer")

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
                    role: metrics.load_states(p, prompt + answer[:-1]) for role, p in paths.items()
                }
            with timing.stages.timed("compare math"):
                return (
                    self.compare_on[dev]
                    .submit(metrics.compare, states, self.heads[dev], answer, len(prompt), dev)
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


def load_tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer  # the GPU box's, like torch

    return AutoTokenizer.from_pretrained(model_dir)


def _render(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Prompt token ids with the teacher's chat template and its default thinking mode."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def fits(tokenizer: Any) -> Callable[[Sample], bool]:
    """Whether a sample's rendered prompt leaves the whole answer budget, for the draw. A byte-level
    token covers at least one UTF-8 byte, so a prompt that short in bytes fits without rendering."""
    limit = config.MAX_MODEL_LEN - config.MAX_TOKENS
    frame = len(_render(tokenizer, [{"role": "user", "content": ""}]))

    def check(sample: Sample) -> bool:
        size = sum(len(m["content"].encode()) for m in sample.messages)
        return size + frame * len(sample.messages) <= limit or (
            len(_render(tokenizer, sample.messages)) <= limit
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
