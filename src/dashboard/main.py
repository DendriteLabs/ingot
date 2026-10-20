"""The dashboard's data. `ingot-dashboard` keeps the site bucket's api/ current: state.json and
each finished eval's page document and artifacts, which the static site at the bucket's domain
reads. `ingot-dashboard-preview` serves the same documents locally with web/ and llms.txt
(/api/state.json rebuilt at most every CACHE_S) to look at the site against a local database."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from datetime import UTC, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from loguru import logger as log

from common import db
from common.config import REPO_ROOT, ChainSpec, Settings, chain_spec, r2_settings, settings
from common.loop import forever
from common.models import bucket_keys, r2_client
from dashboard import config, state

WEB_DIR = REPO_ROOT / "web"


EVAL_DOC = re.compile(r"/api/evals/(\d+)\.json")
EVAL_FILE = re.compile(r"/api/evals/(\d+)/(" + "|".join(map(re.escape, state.ARTIFACTS)) + ")")


class StateCache:
    """One connection, one query at a time; every request inside CACHE_S gets the same state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.link = db.Link()
        self.body, self.built = b"", 0.0

    def query(self, fn: Any) -> Any:
        with self.lock:
            return fn(self.link.conn())

    def get(self) -> bytes:
        if time.monotonic() - self.built > config.CACHE_S:
            doc = self.query(lambda conn: state.build(conn, s=settings(), spec=chain_spec()))
            self.body = _dumps(doc)
            self.built = time.monotonic()
        return self.body


def _json_default(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class Handler(SimpleHTTPRequestHandler):
    cache: StateCache

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/state.json":
            try:
                self._send(self.cache.get(), "application/json")
            except Exception:
                log.exception("building state failed")
                self.cache.link.close()
                self.send_error(503, "state unavailable")
            return
        if path == "/llms.txt":
            self._send((REPO_ROOT / "llms.txt").read_bytes(), "text/plain; charset=utf-8")
            return
        if match := EVAL_DOC.fullmatch(path):
            eval_id = int(match[1])
            doc = self.cache.query(
                lambda conn: state.eval_detail(conn, eval_id, s=settings(), spec=chain_spec())
            )
            if doc is None:
                self.send_error(404, "no such eval")
            else:
                self._send(_dumps(doc), "application/json")
            return
        if match := EVAL_FILE.fullmatch(path):
            self._send_artifact(int(match[1]), match[2])
            return
        super().do_GET()

    def _send_artifact(self, eval_id: int, name: str) -> None:
        row = self.cache.query(
            lambda conn: conn.execute(
                "SELECT s.hotkey FROM evals e JOIN submissions s ON s.id = e.submission_id"
                " WHERE e.id = %s",
                (eval_id,),
            ).fetchone()
        )
        file = row and state.artifact_dir(settings(), row["hotkey"], eval_id) / name
        if not file or not file.exists():
            self.send_error(404, "no such artifact")
            return
        self._send(file.read_bytes(), state.ARTIFACTS[name])

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # static files too: a cached app.js from before a feed change would render nothing
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class Publisher:
    """Keeps the site bucket's api/ current: state.json whenever its content changes, and once
    for every finished eval its page document with both artifacts. A winning eval waits until
    its king is crowned, so the page can name it; the evaluator uploads summary.json last, so an
    eval waits for it, up to ARTIFACT_GRACE_S. Evals already in the bucket are not sent again."""

    def __init__(
        self, link: db.Link, s3: Any, *, s: Settings, spec: ChainSpec, site: str, artifacts: str
    ) -> None:
        self.link, self.s3, self.s, self.spec = link, s3, s, spec
        self.conn = link.conn()
        self.site, self.artifacts = site, artifacts
        self.state_digest: str | None = None
        self.published = {
            int(m[1]) for k in bucket_keys(site, "api/evals/", s3) if (m := EVAL_ID.fullmatch(k))
        }
        # docs written before _dumps nulled non-finite values hold a bare Infinity/NaN: send again
        self.published -= {
            r["id"]
            for r in self.conn.execute(
                """
                SELECT id FROM evals WHERE 'Infinity'::float8 IN (
                    abs(score_challenger), abs(score_king), abs(dist_challenger), abs(dist_king),
                    abs(noise_kl), abs(noise_dist), abs(margin), abs(dist_margin))
                  OR 'NaN'::float8 IN (score_challenger, score_king, dist_challenger, dist_king,
                    noise_kl, noise_dist, margin, dist_margin)
                """
            ).fetchall()
        }

    def run_once(self) -> bool:
        self.conn = self.link.conn()
        busy = self._state()
        for row in self.conn.execute(
            """
            SELECT e.id, e.finished_at, s.hotkey FROM evals e
            JOIN submissions s ON s.id = e.submission_id
            WHERE e.finished_at IS NOT NULL AND NOT (e.id = ANY(%s))
              AND (e.won IS NOT TRUE
                   OR EXISTS (SELECT 1 FROM kings k WHERE k.submission_id = e.submission_id))
            ORDER BY e.id LIMIT %s
            """,
            (list(self.published), config.EVALS_PER_ROUND),
        ).fetchall():
            busy = self._eval(row) or busy
        return busy

    def _state(self) -> bool:
        doc = state.build(self.conn, s=self.s, spec=self.spec)
        # updated_at changes every build; only a change in what the page shows is worth a write
        digest = hashlib.sha256(_dumps({**doc, "updated_at": None})).hexdigest()
        if digest == self.state_digest:
            return False
        self._put("api/state.json", _dumps(doc), "application/json")
        self.state_digest = digest
        return True

    def _eval(self, row: db.Row) -> bool:
        prefix = f"evals/{row['hotkey']}/{row['id']}/"
        present = set(bucket_keys(self.artifacts, prefix, self.s3)) if self.artifacts else set()
        waited = (datetime.now(UTC) - row["finished_at"]).total_seconds()
        if "summary.json" not in present and waited < config.ARTIFACT_GRACE_S:
            return False
        folder = state.artifact_dir(self.s, row["hotkey"], row["id"])
        folder.mkdir(parents=True, exist_ok=True)
        for name in present & state.ARTIFACTS.keys():
            if not (folder / name).exists():
                self.s3.download_file(self.artifacts, prefix + name, str(folder / name))
        doc = state.eval_detail(self.conn, row["id"], s=self.s, spec=self.spec)
        assert doc is not None, "the eval was just selected"
        self._put(f"api/evals/{row['id']}.json", _dumps(doc), "application/json")
        for name in doc["artifacts"]:
            body = (folder / name).read_bytes()
            self._put(f"api/evals/{row['id']}/{name}", body, state.ARTIFACTS[name])
        self.published.add(row["id"])
        log.info("published eval {} ({})", row["id"], ", ".join(doc["artifacts"]) or "no artifacts")
        return True

    def _put(self, key: str, body: bytes, kind: str) -> None:
        self.s3.put_object(
            Bucket=self.site, Key=key, Body=body, ContentType=kind, CacheControl="no-cache"
        )


EVAL_ID = re.compile(r"(\d+)\.json")


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


def _dumps(doc: Any) -> bytes:
    return json.dumps(_finite(doc), default=_json_default, allow_nan=False).encode()


def preview() -> None:
    Handler.cache = StateCache()
    server = ThreadingHTTPServer(
        (config.HOST, config.PORT), partial(Handler, directory=str(WEB_DIR))
    )
    log.info("dashboard on http://{}:{}", config.HOST, config.PORT)
    server.serve_forever()


def main() -> None:
    r2 = r2_settings()
    if not r2.site_bucket_name:
        raise SystemExit("R2_SITE_BUCKET_NAME is not set")
    publisher = Publisher(
        db.Link(),
        r2_client(),
        s=settings(),
        spec=chain_spec(),
        site=r2.site_bucket_name,
        artifacts=r2.artifacts_bucket_name,
    )
    log.info("publishing the dashboard's data to {}/api/", r2.site_bucket_name)
    forever(publisher.run_once, idle_s=config.PUBLISH_S)
