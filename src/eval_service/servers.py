"""The eval's vLLM servers: one per GPU, each extracting hidden states and accepting forced tokens.
The teacher keeps its servers until it is handed another model; the king's and challenger's are
released after each eval."""

from __future__ import annotations

import contextlib
import fcntl
import heapq
import itertools
import json
import math
import os
import re
import signal
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from collections.abc import Generator, Sequence
from pathlib import Path

import httpx
from loguru import logger as log

from common.config import REPO_ROOT, settings
from eval_service import config, timing

HIDDEN_STATES_DIR = settings().hidden_states_dir
_CACHE_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_ERROR_RE = re.compile(r"\b(\w+(?:Error|Exception): .+)")


class ServerExited(RuntimeError):
    """A vLLM server that exited three times while starting; `reason` is the first error line of
    its last start, the root cause (vLLM ends on a generic "Engine core initialization failed")."""

    def __init__(self, gpu: int, reason: str, log_path: Path) -> None:
        self.gpu, self.reason = gpu, reason
        super().__init__(f"vLLM on GPU {gpu} exited three times: {reason}; see {log_path}")


class Server:
    """One `vllm serve` on one GPU, in its own process group so stopping it stops its workers."""

    def __init__(self, model_dir: Path, gpu: int, share: float) -> None:
        self.gpu, self.url = gpu, f"http://127.0.0.1:{config.BASE_PORT + gpu}"
        self.log = settings().eval_work_dir / "vllm" / f"gpu{gpu}.log"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        states_dir = HIDDEN_STATES_DIR / f"gpu{gpu}"
        states_dir.mkdir(parents=True, exist_ok=True)
        self.command, self.cache_tokens = _command(model_dir, gpu, share, states_dir), 0
        # the last run's log stays beside the new one: a server that died is often looked at only
        # after the next eval started its own
        if self.log.exists():
            self.log.replace(self.log.with_suffix(".prev.log"))
        self.log.write_text("")
        self._launch()

    def _launch(self) -> None:
        self.launched_at = self.log.stat().st_size
        with open(self.log, "a") as out:
            self.process = subprocess.Popen(
                self.command,
                env=_env(self.gpu),
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def wait_ready(self, deadline: float) -> None:
        # Servers cold-compiling the same graph race on the shared compile cache and the losers
        # die; restarted, they find the cache warm. Separate caches per server would instead
        # repeat the whole compile on every GPU.
        for restarts in range(3):
            while time.monotonic() < deadline and self.process.poll() is None:
                with contextlib.suppress(httpx.HTTPError):
                    if httpx.get(f"{self.url}/health", timeout=5).status_code == 200:
                        found = _CACHE_RE.findall(self.log.read_text())
                        self.cache_tokens = int(found[-1].replace(",", ""))
                        return
                time.sleep(10)
            if self.process.poll() is None:
                raise TimeoutError(f"vLLM on GPU {self.gpu} not healthy in time; see {self.log}")
            if restarts < 2:
                log.warning("vLLM on GPU {} exited while starting; restarting it", self.gpu)
                self.stop()
                self._launch()
        raise ServerExited(self.gpu, self._first_error(), self.log)

    def _first_error(self) -> str:
        with open(self.log, "rb") as f:
            f.seek(self.launched_at)
            text = f.read().decode(errors="replace")
        found = _ERROR_RE.search(text)
        return found.group(1).strip()[:300] if found else "no error line in its log"

    def alive(self) -> bool:
        return self.process.poll() is None

    def preemptions(self) -> float:
        text = httpx.get(f"{self.url}/metrics", timeout=30).text
        return sum(
            float(line.split()[-1])
            for line in text.splitlines()
            if line.startswith("vllm:num_preemptions_total")
        )

    def stop(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=60)
        # the engine workers outlive the API server for a moment; their memory must be free
        # before the next server starts on this GPU
        for _ in range(60):
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(1)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)


class Admission:
    """Admits requests to a group's servers in line order: the first in line goes to the server
    with the most free budget once it fits there, and the rest wait behind it. Letting whatever
    fits go first starved the long prompts until the end of an eval; forced reads (`first`) go
    ahead of new answers, so a sample whose answer is done is not held behind them."""

    def __init__(self, n_servers: int) -> None:
        self.used = [0] * n_servers
        self.cv = threading.Condition()
        self.line: list[tuple[int, int]] = []
        self.tickets = itertools.count()

    def enter(self, need: int, caps: Sequence[int], first: bool) -> int:
        """Wait for this request's turn and room, then book `need` tokens; the server index."""
        with self.cv:
            ticket = (0 if first else 1, next(self.tickets))
            heapq.heappush(self.line, ticket)
            try:
                self.cv.wait_for(
                    lambda: (
                        self.line[0] == ticket
                        and any(u + need <= c for u, c in zip(self.used, caps, strict=True))
                    )
                )
            except BaseException:
                self.line.remove(ticket)
                heapq.heapify(self.line)
                self.cv.notify_all()
                raise
            heapq.heappop(self.line)
            i = max(range(len(caps)), key=lambda k: caps[k] - self.used[k])
            self.used[i] += need
            # the next in line may fit as well
            self.cv.notify_all()
            return i

    def leave(self, i: int, need: int) -> None:
        with self.cv:
            self.used[i] -= need
            self.cv.notify_all()


class Group:
    """One model on several GPUs. A request goes to the server with the most free budget and is
    admitted only while every admitted request fits that server's cache at full length: vLLM
    recomputes a preempted request in one prefill pass, silently turning decode rows into
    prefill rows."""

    def __init__(
        self, model_dir: Path, gpus: Sequence[int], share: float, role: str = "model"
    ) -> None:
        self.model_dir, self.role = model_dir, role
        self.servers = [Server(model_dir, gpu, share) for gpu in gpus]
        self.admission = Admission(len(self.servers))

    def wait_ready(self, deadline: float) -> None:
        for server in self.servers:
            server.wait_ready(deadline)

    def alive(self) -> bool:
        return all(server.alive() for server in self.servers)

    def preemptions(self) -> float:
        return sum(server.preemptions() for server in self.servers)

    def stop(self) -> None:
        for server in self.servers:
            server.stop()

    def complete(
        self, prompt: list[int], max_tokens: int, seed: int, force: list[int] | None = None
    ) -> tuple[list[int], str]:
        """Answer token ids and the hidden-state file of one request."""
        body = {
            "model": "candidate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "seed": seed,
            **config.SAMPLING,
            "return_token_ids": True,
            "kv_transfer_params": {"include_output_tokens": True},
        }
        if force is not None:
            # max_tokens ends a forced read; stopping at the server's EOS instead would depend
            # on the tokenizer the submission ships (none: an empty one whose EOS is id 0, "!")
            body["vllm_xargs"] = {"force_token_ids": force}
            body["ignore_eos"] = True
        with (
            self._hold(len(prompt) + max_tokens, first=force is not None) as server,
            timing.stages.timed(f"{self.role} request"),
        ):
            resp = httpx.post(f"{server.url}/v1/completions", json=body, timeout=None)
        resp.raise_for_status()
        out = resp.json()
        return out["choices"][0]["token_ids"], out["kv_transfer_params"]["hidden_states_path"]

    @contextlib.contextmanager
    def _hold(self, n_tokens: int, *, first: bool) -> Generator[Server]:
        # whole 800-token attention blocks, plus one block for the request's GDN state
        need = (math.ceil(n_tokens / 800) + 1) * 800
        caps = [int(0.9 * server.cache_tokens) for server in self.servers]
        if need > max(caps):
            raise ValueError(f"a request of {n_tokens} tokens can never fit {max(caps)}")
        with timing.stages.timed(f"{self.role} wait"):
            i = self.admission.enter(need, caps, first)
        try:
            yield self.servers[i]
        finally:
            self.admission.leave(i, need)


class LlamaServer:
    """One llama-states process on one GPU: the gguf model reads forced tokens and writes the
    hidden states of LAYERS as a vLLM server's states file. It takes one request at a time."""

    def __init__(self, model_dir: Path, gpu: int, index: int = 0) -> None:
        (self.model,) = model_dir.glob("*.gguf")
        self.gpu, self.lock = gpu, threading.Lock()
        self.dir = HIDDEN_STATES_DIR / f"gpu{gpu}"
        self.dir.mkdir(parents=True, exist_ok=True)
        name = f"gpu{gpu}.log" if index == 0 else f"gpu{gpu}-{index}.log"
        self.log = settings().eval_work_dir / "llama" / name
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists():
            self.log.replace(self.log.with_suffix(".prev.log"))
        layers = ",".join(str(layer - 1) for layer in config.LAYERS)  # block il is layer il + 1
        self.process = subprocess.Popen(
            [
                str(REPO_ROOT / config.LLAMA_STATES),
                "-m",
                str(self.model),
                "--layers",
                layers,
                "-c",
                str(config.MAX_MODEL_LEN),
                "-b",
                str(config.LLAMA_BATCH),
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(self.log, "w"),  # noqa: SIM115 - the child keeps it
            text=True,
            start_new_session=True,
        )

    def wait_ready(self, deadline: float) -> None:
        """The model is loaded when the process says so. Its first line is read on a thread, so
        a file that hangs the fork while loading fails at the deadline (the process is killed)
        instead of wedging the eval."""
        line = self._readline(deadline, "not loaded before the deadline")
        try:
            ready = bool(line) and json.loads(line).get("ready") is True
        except json.JSONDecodeError:
            ready = False
        if not ready:
            raise ServerExited(self.gpu, self._first_error(), self.log)

    def _first_error(self) -> str:
        text = self.log.read_text(errors="replace")
        found = re.search(r"(?im)^.*\b(error|failed|cannot)\b.*$", text)
        return found.group(0).strip()[:300] if found else "no error line in its log"

    def alive(self) -> bool:
        return self.process.poll() is None

    def _readline(self, deadline: float, late: str) -> str:
        """The process's next stdout line, read on a thread so a hung process cannot block past
        `deadline`: then it is stopped and ServerExited names its GPU ("" when it exited)."""
        out = self.process.stdout
        assert out
        got: list[str] = []
        reader = threading.Thread(target=lambda: got.append(out.readline()), daemon=True)
        reader.start()
        reader.join(max(0.0, deadline - time.monotonic()))
        if reader.is_alive():
            self.stop()
            raise ServerExited(self.gpu, late, self.log)
        return got[0] if got else ""

    def read(self, tokens: list[int], first: int) -> str:
        """One forced read. A process that dies or hangs raises ServerExited for its GPU, so the
        eval knows whose model failed (a challenger's: it does not run; a king's: not the
        challenger's fault)."""
        path = str(self.dir / f"{uuid.uuid4().hex}.safetensors")
        request = {"tokens": tokens, "first": first, "out": path}
        assert self.process.stdin
        # generous for a 32k-token read at a few thousand tokens per second
        budget = config.LLAMA_READ_S + len(tokens) / 1000 * config.LLAMA_S_PER_1K
        with self.lock:
            deadline = time.monotonic() + budget
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                raise ServerExited(self.gpu, self._first_error(), self.log) from None
            line = self._readline(deadline, f"a read of {len(tokens)} tokens did not finish")
        if not line:
            raise ServerExited(self.gpu, self._first_error(), self.log)
        reply = json.loads(line)
        if not reply["ok"]:
            discard(path)
            raise RuntimeError(f"llama-states on GPU {self.gpu}: {reply['error']}")
        return path

    def stop(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=60)
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)


class LlamaGroup:
    """A gguf model on several GPUs, LLAMA_PER_GPU llama-states each. It only reads: the answer is
    always the teacher's, so a request is a forced read and returns exactly the forced answer."""

    def __init__(self, model_dir: Path, gpus: Sequence[int], role: str = "model") -> None:
        self.model_dir, self.role = model_dir, role
        self.servers = [
            LlamaServer(model_dir, gpu, k) for gpu in gpus for k in range(config.LLAMA_PER_GPU)
        ]
        self.waiting = [0] * len(self.servers)
        self.cv = threading.Condition()

    def wait_ready(self, deadline: float) -> None:
        for server in self.servers:
            server.wait_ready(deadline)

    def alive(self) -> bool:
        return all(server.alive() for server in self.servers)

    def preemptions(self) -> float:
        return 0.0  # a read is one pass over its own context; nothing is ever recomputed

    def stop(self) -> None:
        for server in self.servers:
            server.stop()

    def complete(
        self, prompt: list[int], max_tokens: int, seed: int, force: list[int] | None = None
    ) -> tuple[list[int], str]:
        if force is None or len(force) != len(prompt) + max_tokens:
            raise ValueError(f"the {self.role} runs in llama.cpp and only reads a forced answer")
        with self.cv:
            i = min(range(len(self.servers)), key=self.waiting.__getitem__)
            self.waiting[i] += 1
        try:
            with timing.stages.timed(f"{self.role} request"):
                # the last answer token is never read back in: rows are prompt + answer[:-1]
                path = self.servers[i].read(force[:-1], len(prompt) - 1)
        finally:
            with self.cv:
                self.waiting[i] -= 1
        return force[len(prompt) :], path


class RemoteGroup:
    """The teacher served by another box (ingot-teacher-server, through a tunnel): its answers come
    over HTTP and each request's hidden states are downloaded into the local states dir, where the
    comparison reads them as it reads a local server's. `url` may list several comma-separated
    tunnels to the same server; each request and its download take the next one in turn."""

    def __init__(self, url: str, model_dir: Path) -> None:
        self.urls = [u.strip().rstrip("/") for u in url.split(",") if u.strip()]
        self.url, self.model_dir = self.urls[0], model_dir
        self._turn = itertools.cycle(self.urls)
        self._lock = threading.Lock()
        self.dir = HIDDEN_STATES_DIR / "remote"
        self.dir.mkdir(parents=True, exist_ok=True)

    def wait_ready(self, deadline: float) -> None:
        while time.monotonic() < deadline:
            if self.alive():
                return
            time.sleep(10)
        raise TimeoutError(f"the remote teacher at {self.url} is not healthy")

    def alive(self) -> bool:
        """Every tunnel answers: a dead one would fail every request sent its way."""
        with contextlib.suppress(httpx.HTTPError):
            return all(httpx.get(f"{u}/health", timeout=5).status_code == 200 for u in self.urls)
        return False

    def preemptions(self) -> float:
        return float(httpx.get(f"{self.url}/preemptions", timeout=30).json()["preemptions"])

    def stop(self) -> None:
        pass

    def complete(
        self, prompt: list[int], max_tokens: int, seed: int, force: list[int] | None = None
    ) -> tuple[list[int], str]:
        body = {"prompt": prompt, "max_tokens": max_tokens, "seed": seed, "force": force}
        with self._lock:
            url = next(self._turn)
        with timing.stages.timed("teacher request"):
            resp = httpx.post(f"{url}/complete", json=body, timeout=None)
        resp.raise_for_status()
        out = resp.json()
        path = str(self.dir / f"{uuid.uuid4().hex}.safetensors")
        Path(path + ".lock").touch()
        size = 0
        try:
            with (
                timing.stages.timed("teacher download"),
                httpx.stream("GET", f"{url}/states/{out['states']}", timeout=None) as r,
            ):
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
                        size += len(chunk)
        except BaseException:
            discard(path)
            raise
        timing.stages.add("teacher states MB", size / 1e6)
        return out["token_ids"], path


def discard(path: str) -> None:
    """Delete a request's state file, after the server finished writing it: deleting earlier
    leaves an orphan the server writes afterwards, which fills /dev/shm."""
    with contextlib.suppress(FileNotFoundError), open(path + ".lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
    for name in (path + ".lock", path):
        with contextlib.suppress(FileNotFoundError):
            os.remove(name)


def sweep(older_than_s: float = 0.0) -> int:
    """Delete the state files (and lone locks) under HIDDEN_STATES_DIR untouched for `older_than_s`
    seconds: what a killed process, or an eval that never fetched them, left behind. The count."""
    cutoff = time.time() - older_than_s
    paths = {
        p.with_suffix("") if p.suffix == ".lock" else p
        for p in HIDDEN_STATES_DIR.rglob("*.safetensors*")
    }
    swept = 0
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            times = [p.stat().st_mtime for p in (path, Path(f"{path}.lock")) if p.exists()]
            if times and max(times) < cutoff:
                discard(str(path))
                swept += 1
    return swept


Groups = Group | LlamaGroup | RemoteGroup
_groups: dict[str, Groups] = {}


def serve(models: dict[str, Path]) -> dict[str, Groups]:
    """Groups serving `models` (role → model dir), healthy. A role whose model changed or whose
    server died is restarted; the others keep running. With INGOT_TEACHER_URL set the teacher is
    another box's and takes no GPU here."""
    remote_teacher = settings().teacher_url
    started = time.monotonic()
    for role, model_dir in models.items():
        group = _groups.get(role)
        if group and group.model_dir == model_dir and group.alive():
            log.info("the {} keeps its servers: {}", role, model_dir)
            continue
        if group:
            group.stop()
        if role == "teacher" and remote_teacher:
            log.info("using the teacher at {}: {}", remote_teacher, model_dir)
            _groups[role] = RemoteGroup(remote_teacher, model_dir)
            continue
        gpus = settings().gpus(role)
        if not gpus:
            raise RuntimeError(
                f"no GPUs for the {role}: set INGOT_{role.upper()}_GPUS"
                + (" or INGOT_TEACHER_URL" if role == "teacher" else "")
            )
        log.info("starting the {} on GPUs {}: {}", role, gpus, model_dir)
        if any(model_dir.glob("*.gguf")):
            _groups[role] = LlamaGroup(model_dir, gpus, role)
        else:
            _groups[role] = Group(model_dir, gpus, config.MEMORY_SHARE[role], role)
    deadline = time.monotonic() + config.SERVER_START_S
    # waited in turn: a role's time is when it was found ready, at most later than it became so
    for role in models:
        _groups[role].wait_ready(deadline)
        log.info("the {} is ready, {:.0f}s into serve", role, time.monotonic() - started)
    return {role: _groups[role] for role in models}


def release(*roles: str) -> None:
    """Stops these roles' servers, so their GPUs are free between evals."""
    for role in roles:
        if group := _groups.pop(role, None):
            group.stop()


def stop_all() -> None:
    for group in _groups.values():
        group.stop()
    _groups.clear()


def _command(model_dir: Path, gpu: int, share: float, states_dir: Path) -> list[str]:
    extract = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": config.LAYERS}},
    }
    connector = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(states_dir)},
    }
    return [
        str(Path(sys.executable).with_name("vllm")),
        "serve",
        str(model_dir),
        "--served-model-name",
        "candidate",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.BASE_PORT + gpu),
        "--gpu-memory-utilization",
        str(share),
        "--generation-config",
        "vllm",
        "--language-model-only",
        "--max-model-len",
        str(config.MAX_MODEL_LEN),
        # the extractor's supported setup: a whole prompt is one prefill pass, and every row is
        # computed for its own request (prefix-cache reuse is not bit-exact)
        "--max-num-batched-tokens",
        str(config.MAX_MODEL_LEN),
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        "64",
        "--mamba-ssm-cache-dtype",
        "float32",
        # sm_120's FlashInfer GDN prefill kernel is about 1% off the reference
        "--gdn-prefill-backend",
        "triton",
        "--logits-processors",
        "eval_service.forcer:TrajectoryForcer",
        "--speculative-config",
        json.dumps(extract),
        "--kv-transfer-config",
        json.dumps(connector),
    ]


def _env(gpu: int) -> dict[str, str]:
    # The CUDA toolkit comes from the pip wheels inside the venv; FlashInfer's and Triton's JIT
    # (nvcc, ninja, the linker) must be pointed at it.
    cu = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    libs = f"{cu}/lib:{cu}/lib64"
    cache = settings().eval_work_dir / "compile-cache"
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_HOME": str(cu),
        "CUDA_PATH": str(cu),
        "PATH": f"{cu}/bin:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{libs}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "LIBRARY_PATH": f"{libs}:{os.environ.get('LIBRARY_PATH', '')}",
        "NVCC_APPEND_FLAGS": "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
        # kept across restarts: a cold compile adds about ten minutes to a server start
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "VLLM_ENGINE_READY_TIMEOUT_S": str(config.SERVER_START_S),
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
