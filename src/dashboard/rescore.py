"""Every duel scored again under the current rule from its per-sample records, so the kings
chart chains one rule across evals that ran under earlier ones. A duel is scored once, from the
local copy of its samples.jsonl (fetched from the artifacts bucket when the copy is missing), and
the result kept in a JSON file beside the copies."""

from __future__ import annotations

import json
import threading
from typing import Any

from loguru import logger as log

from common.config import Settings
from dashboard import state


class Rescorer:
    def __init__(self, s: Settings, s3: Any = None, artifacts: str = "") -> None:
        self.s, self.s3, self.artifacts = s, s3, artifacts
        self.path = s.artifacts_dir / "rescored.json"
        self.lock = threading.Lock()
        self.cache: dict[int, list[float] | None] = {}
        if self.path.exists():
            self.cache = {int(k): v for k, v in json.loads(self.path.read_text()).items()}

    def score(self, hotkey: str, eval_id: int) -> tuple[float, float] | None:
        """The duel's (king, challenger) under the current rule: from the cache, else from its
        local samples; None when neither has it."""
        if eval_id in self.cache:
            v = self.cache[eval_id]
            return (v[0], v[1]) if v else None
        samples = state.artifact_dir(self.s, hotkey, eval_id) / "samples.jsonl"
        return self._score(eval_id, samples) if samples.exists() else None

    def fill(self, duels: list[tuple[str, int]], limit: int) -> int:
        """Score up to `limit` of `duels` ((hotkey, eval id), in the order they matter) that the
        cache lacks, fetching their samples; a duel whose samples are not in the bucket is
        remembered as unscorable. The number scored."""
        done = 0
        for hotkey, eval_id in duels:
            if done >= limit:
                break
            if eval_id in self.cache:
                continue
            samples = state.artifact_dir(self.s, hotkey, eval_id) / "samples.jsonl"
            if not samples.exists():
                if self.s3 is None or not self.artifacts:
                    continue
                samples.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self.s3.download_file(
                        self.artifacts, f"evals/{hotkey}/{eval_id}/samples.jsonl", str(samples)
                    )
                except Exception as exc:
                    log.info("eval {} has no samples to rescore: {}", eval_id, exc)
                    self._remember(eval_id, None)
                    done += 1
                    continue
            self._score(eval_id, samples)
            done += 1
        return done

    def _score(self, eval_id: int, samples: Any) -> tuple[float, float] | None:
        records = [json.loads(line) for line in samples.read_text().split("\n") if line.strip()]
        scored = state.token_weighted(records)
        self._remember(eval_id, list(scored) if scored else None)
        return scored

    def _remember(self, eval_id: int, value: list[float] | None) -> None:
        with self.lock:
            self.cache[eval_id] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({str(k): v for k, v in self.cache.items()}))
