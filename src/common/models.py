"""Where model bytes come from and go to: the private R2 bucket, Hugging Face, the local cache.

A model uri is either `s3://<bucket>/<prefix>@sha256:<digest>` (a verified private upload)
or `<namespace>/<repo>[@<revision>]` on Hugging Face (genesis, the teacher, mirrored kings).
"""

from __future__ import annotations

import hashlib
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

from common.config import Quant, r2_settings, settings
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
    repo, revision, file = _hub(uri)
    return snapshot(repo, revision, allow_patterns=file)


def genesis_uri(quant: Quant) -> str:
    """repo@revision of a quant's genesis, with #file when it is one file of its repo."""
    uri = f"{quant.genesis_repo}@{quant.genesis_revision}"
    return f"{uri}#{quant.genesis_file}" if quant.genesis_file else uri


def list_files(uri: str) -> list[str]:
    """File names of the model at `uri` without downloading it; the pre-download file check."""
    match = _PRIVATE_RE.match(uri)
    if match:
        return bucket_keys(match["bucket"], match["prefix"] + "/")
    from huggingface_hub import HfApi

    repo, revision, file = _hub(uri)
    files = HfApi(token=_token()).list_repo_files(repo, revision=revision)
    return [f for f in files if file is None or f == file]


def _hub(uri: str) -> tuple[str, str | None, str | None]:
    """repo@revision, or repo@revision#file for a model that is one file of its repo (a gguf
    genesis next to its other quantizations)."""
    rest, _, file = uri.partition("#")
    repo, _, revision = rest.partition("@")
    return repo, revision or None, file or None


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


def publish(
    model_dir: Path, repo_id: str, card: str, pair: Quant, extra: dict[str, Path] | None = None
) -> str:
    """Mirror a king to a public Hugging Face repo, completed with the files of its quant's
    genesis it was not allowed to ship (pair.publish_files) and `extra` (name in the repo → local
    file): one commit after the repo's initial one. A gguf king's one file is named after the
    repo. Returns the repo url."""
    from huggingface_hub import CommitOperationAdd, HfApi

    files = {name: str(path) for name, path in _mirrored(model_dir, repo_id, pair).items()}
    for name in pair.publish_files:
        files[name] = str(hub_file(pair.genesis_repo, name, pair.genesis_revision))
    files |= {name: str(path) for name, path in (extra or {}).items()}
    operations = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=f) for n, f in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    api = HfApi(token=_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.create_commit(repo_id=repo_id, operations=operations, commit_message="Mirror Ingot king")
    log.info("published {} to {} ({} files)", model_dir, repo_id, len(operations))
    return f"https://huggingface.co/{repo_id}"


def publish_dir(pack_dir: Path, repo_id: str, card: str, pair: Quant) -> str:
    """publish() for a directory that is the whole upload, a king's MLX pack: every file of
    pack_dir, the genesis files of pair.publish_files and the card, in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    files = {name: str(path) for name, path in _mirrored(pack_dir, repo_id, pair).items()}
    for name in pair.publish_files:
        files[name] = str(hub_file(pair.genesis_repo, name, pair.genesis_revision))
    operations = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=f) for n, f in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    api = HfApi(token=_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.create_commit(repo_id=repo_id, operations=operations, commit_message="Mirror Ingot king")
    log.info("published {} to {} ({} files)", pack_dir, repo_id, len(operations))
    return f"https://huggingface.co/{repo_id}"


def update_card(repo_id: str, line: str) -> None:
    """Append `line` to a mirror's README.md unless it is there already: one commit of that file."""
    from huggingface_hub import CommitOperationAdd, HfApi

    text = hub_file(repo_id, "README.md").read_text()
    if line.strip() in text:
        return
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=(text + line).encode())
    ]
    HfApi(token=_token()).create_commit(
        repo_id=repo_id, operations=operations, commit_message="Link the MLX pack"
    )
    log.info("linked the MLX pack from the card of {}", repo_id)


def _mirrored(model_dir: Path, repo_id: str, pair: Quant) -> dict[str, Path]:
    """name in the mirror → local file, for the files of the king's own upload."""
    files: dict[str, Path] = {}
    for path in sorted(model_dir.rglob("*")):
        name = path.relative_to(model_dir).as_posix()
        if path.is_file() and not any(fnmatch(name, p) for p in _UPLOAD_IGNORE):
            if pair.format == "gguf" and name.endswith(".gguf"):
                name = gguf_name(repo_id.rsplit("/", 1)[-1], pair.king_packing)
            files[name] = path
    return files


def gguf_name(repo_name: str, packing: str | None) -> str:
    """A king's gguf in its mirror, named for its packing in place of the quant label
    (Ingot-IV-27B-Q2 -> Ingot-IV-27B-PQ2_0.gguf); without one, for the repo."""
    return f"{repo_name.rsplit('-', 1)[0]}-{packing}.gguf" if packing else f"{repo_name}.gguf"


def mlx_repo_name(repo_name: str) -> str:
    """A king's MLX pack repo, named beside its gguf repo (Ingot-IV-27B-Q2 ->
    Ingot-IV-27B-MLX-2bit)."""
    return f"{repo_name.rsplit('-', 1)[0]}-MLX-2bit"


def mirror_matches(model_dir: Path, repo_id: str, pair: Quant) -> bool:
    """Whether the Hugging Face mirror holds every file of `model_dir` byte for byte: one
    model_info call, each file compared by size and by its lfs sha256 or git blob sha1."""
    from huggingface_hub import HfApi

    info = HfApi(token=_token()).model_info(repo_id, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings or []}
    for name, path in _mirrored(model_dir, repo_id, pair).items():
        sibling = remote.get(name)
        size = path.stat().st_size
        if sibling is None:
            log.warning("mirror {} lacks {}", repo_id, name)
            return False
        if sibling.lfs is not None:
            same = sibling.lfs.size == size and sibling.lfs.sha256 == _sha256(path)
        else:
            same = sibling.size == size and sibling.blob_id == _git_blob_sha1(path)
        if not same:
            log.warning("mirror {} differs at {}", repo_id, name)
            return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


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
    # the marker is written only after the whole folder re-hashed to the digest
    if marker.exists() and dest.is_dir() and any(dest.iterdir()):
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
