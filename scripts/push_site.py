"""Upload the website to the R2 bucket named by R2_SITE_BUCKET_NAME.

    uv run scripts/push_site.py

Pushes web/ to the bucket root and the repo's llms.txt beside it. Only uploads: other objects in
the bucket are never touched. api/state.json is not part of this push; the backend writes it.

A push shows up on a browser's next load, plain reload included. The HTML is never stored, and
every script and stylesheet it loads, and every module those import, is referenced as
`<path>?v=<version>`, the version a hash of the whole site: a changed site is all new URLs, which
no cache has, while an unchanged one keeps its URLs and its cache. Images and the rest are cached
for a minute.
"""

from __future__ import annotations

import hashlib
import mimetypes
import re
from pathlib import PurePosixPath

from common.config import REPO_ROOT, r2_settings
from common.models import r2_client

WEB = REPO_ROOT / "web"
TYPES = {".js": "text/javascript", ".svg": "image/svg+xml", ".txt": "text/plain"}
# a local script or stylesheet in an HTML attribute; a relative module import in JS
HTML_REF = re.compile(r'((?:src|href)=")([^"#:?]+\.(?:js|css))(")')
JS_IMPORT = re.compile(r'((?:from|import)\s*\(?\s*")(\.{1,2}/[^"?]+\.js)(")')


def version(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for key in sorted(files):
        digest.update(key.encode() + b"\0" + files[key])
    return digest.hexdigest()[:10]


def stamp(key: str, body: bytes, v: str) -> bytes:
    """`body` with its local script, stylesheet and module references at version `v`."""
    pattern = HTML_REF if key.endswith(".html") else JS_IMPORT if key.endswith(".js") else None
    if pattern is None:
        return body
    return pattern.sub(lambda m: f"{m[1]}{m[2]}?v={v}{m[3]}", body.decode()).encode()


def cache_control(key: str) -> str:
    if key.endswith(".html"):
        return "no-store"
    if key.endswith((".css", ".js")):
        return "no-cache"
    return "public, max-age=60"


def site() -> dict[str, bytes]:
    files = {
        p.relative_to(WEB).as_posix(): p.read_bytes() for p in sorted(WEB.rglob("*")) if p.is_file()
    }
    files["llms.txt"] = (REPO_ROOT / "llms.txt").read_bytes()
    return files


def main() -> None:
    bucket = r2_settings().site_bucket_name
    if not bucket:
        raise SystemExit("R2_SITE_BUCKET_NAME is not set")
    files = site()
    v = version(files)
    s3 = r2_client()
    for key, body in files.items():
        suffix = PurePosixPath(key).suffix
        kind = TYPES.get(suffix) or mimetypes.guess_type(key)[0] or "application/octet-stream"
        text = kind.startswith("text/") or kind in ("image/svg+xml", "application/json")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=stamp(key, body, v),
            ContentType=f"{kind}; charset=utf-8" if text else kind,
            CacheControl=cache_control(key),
        )
        print(f"{key:<24} {kind}")
    print(f"pushed {len(files)} files to {bucket}, version {v}")


if __name__ == "__main__":
    main()
