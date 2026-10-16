"""Upload the website to the R2 bucket named by R2_SITE_BUCKET_NAME.

    uv run scripts/push_site.py

Pushes web/ to the bucket root and the repo's llms.txt beside it. Only uploads: other objects in
the bucket are never touched. api/state.json is not part of this push; the backend writes it.
Browsers revalidate the HTML, CSS and JS on every visit, so a push shows up on the next load;
images and the rest are cached for a minute.
"""

from __future__ import annotations

import mimetypes

from common.config import REPO_ROOT, r2_settings
from common.models import r2_client

WEB = REPO_ROOT / "web"
TYPES = {".js": "text/javascript", ".svg": "image/svg+xml", ".txt": "text/plain"}
FRESH = (".html", ".css", ".js")  # always revalidated


def main() -> None:
    bucket = r2_settings().site_bucket_name
    if not bucket:
        raise SystemExit("R2_SITE_BUCKET_NAME is not set")
    files = {p.relative_to(WEB).as_posix(): p for p in sorted(WEB.rglob("*")) if p.is_file()}
    files["llms.txt"] = REPO_ROOT / "llms.txt"
    s3 = r2_client()
    for key, path in files.items():
        kind = (
            TYPES.get(path.suffix)
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        text = kind.startswith("text/") or kind in ("image/svg+xml", "application/json")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=path.read_bytes(),
            ContentType=f"{kind}; charset=utf-8" if text else kind,
            CacheControl="no-cache" if key.endswith(FRESH) else "public, max-age=60",
        )
        print(f"{key:<24} {kind}")
    print(f"pushed {len(files)} files to {bucket}")


if __name__ == "__main__":
    main()
