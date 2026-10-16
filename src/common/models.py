"""Where model bytes come from and go to: the private R2 bucket, Hugging Face, the local cache.

A model uri is either `s3://<bucket>/<prefix>@sha256:<digest>` (a verified private upload)
or `<namespace>/<repo>[@<revision>]` on Hugging Face (genesis, the teacher, mirrored kings).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any

from loguru import logger as log

from common.config import chain_spec, r2_settings, settings
from private_store.digests import verify_snapshot

_PRIVATE_RE = re.compile(
    r"^s3://(?P<bucket>[^/]+)/(?P<prefix>[^@]+?)/?@sha256:(?P<digest>[0-9a-f]{64})$"
)
_UPLOAD_IGNORE = [".*"]


def fetch(uri: str, cache_dir: Path) -> Path:
    """Local directory holding the model at `uri`, downloading and verifying it if needed."""
    match = _PRIVATE_RE.match(uri)
    if match:
        return _fetch_private(match["bucket"], match["prefix"] + "/", match["digest"], cache_dir)
    repo, _, revision = uri.partition("@")
    return snapshot(repo, revision or None)


def list_files(uri: str) -> list[str]:
    """File names of the model at `uri` without downloading it; the pre-download file check."""
    match = _PRIVATE_RE.match(uri)
    if match:
        return bucket_keys(match["bucket"], match["prefix"] + "/")
    from huggingface_hub import HfApi

    repo, _, revision = uri.partition("@")
    return list(HfApi(token=_token()).list_repo_files(repo, revision=revision or None))


def safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Tensor name → {dtype, shape, data_offsets} from one shard's header; no weights are read."""
    with path.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
    header.pop("__metadata__", None)
    return header


def snapshot(
    repo: str,
    revision: str | None = None,
    *,
    repo_type: str | None = None,
    allow_patterns: str | list[str] | None = None,
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo,
            revision=revision,
            repo_type=repo_type,
            token=_token(),
            cache_dir=settings().hf_cache_dir,
            allow_patterns=allow_patterns,
        )
    )


def hub_file(repo: str, filename: str, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            token=_token(),
            cache_dir=settings().hf_cache_dir,
        )
    )


def hub_layout(repo: str, revision: str | None = None) -> dict[str, tuple[str, tuple[int, ...]]]:
    """tensor name → (dtype, shape) for every safetensors tensor, read from the headers only."""
    from huggingface_hub import HfApi

    meta = HfApi(token=_token()).get_safetensors_metadata(repo, revision=revision)
    return {
        name: (info.dtype, tuple(info.shape))
        for file_meta in meta.files_metadata.values()
        for name, info in file_meta.tensors.items()
    }


def publish(model_dir: Path, repo_id: str, card: str) -> str:
    """Mirror a king to a public Hugging Face repo, completed with the genesis tokenizer and
    configs it was not required to ship: one commit after the repo's initial one. Returns the
    repo url."""
    from huggingface_hub import CommitOperationAdd, HfApi

    files: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        name = path.relative_to(model_dir).as_posix()
        if path.is_file() and not any(fnmatch(name, p) for p in _UPLOAD_IGNORE):
            files[name] = str(path)
    spec = chain_spec()
    for name in spec.genesis_files:
        files[name] = str(hub_file(spec.genesis_repo, name, spec.genesis_revision))
    operations = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=f) for n, f in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    api = HfApi(token=_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.create_commit(repo_id=repo_id, operations=operations, commit_message="Mirror Ingot king")
    log.info("published {} to {} ({} files)", model_dir, repo_id, len(operations))
    return f"https://huggingface.co/{repo_id}"


def upload_artifact(key: str, body: bytes) -> str | None:
    bucket = r2_settings().artifacts_bucket_name
    if not bucket:
        return None
    try:
        r2_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson" if key.endswith(".jsonl") else "application/json",
        )
    except Exception:
        log.exception("artifact upload {} failed", key)
        return None
    return f"s3://{bucket}/{key}"


def _token() -> str | None:
    return settings().hf_token or None


@lru_cache(maxsize=1)
def r2_client() -> Any:
    import boto3
    from botocore.config import Config

    r2 = r2_settings()
    return boto3.client(
        "s3",
        endpoint_url=r2.endpoint,
        aws_access_key_id=r2.access_key_id,
        aws_secret_access_key=r2.secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )


def bucket_keys(bucket: str, prefix: str, s3: Any = None) -> list[str]:
    """Every object under `prefix`, as paths relative to it; `s3` defaults to the R2 client."""
    s3 = s3 or r2_client()
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for item in page.get("Contents", []):
            rel = item["Key"][len(prefix) :]
            if rel and not rel.endswith("/"):
                keys.append(rel)
        if not page.get("IsTruncated"):
            return keys
        token = page.get("NextContinuationToken")


def _fetch_private(bucket: str, prefix: str, digest: str, cache_dir: Path) -> Path:
    from boto3.s3.transfer import TransferConfig

    dest = cache_dir / digest
    marker = cache_dir / f"{digest}.verified"
    if marker.exists() and any(dest.glob("*.safetensors")):
        return dest
    shutil.rmtree(dest, ignore_errors=True)  # an unverified partial starts over
    dest.mkdir(parents=True)
    root = dest.resolve()
    config = TransferConfig(max_concurrency=16)
    for name in bucket_keys(bucket, prefix):
        target = (dest / name).resolve()
        if not str(target).startswith(str(root) + os.sep):
            raise ValueError(f"object key escapes the model directory: {name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        r2_client().download_file(bucket, prefix + name, str(target), Config=config)
    verify_snapshot(dest, digest)  # the private store has no server-side revisions: re-hash
    marker.write_text("")
    log.info("fetched s3://{}/{} → {}", bucket, prefix, dest)
    return dest
