from __future__ import annotations

import concurrent.futures
import hashlib
from pathlib import Path


class ArtifactIntegrityError(RuntimeError):
    pass


def model_digest_from_inventory(files: list[tuple[str, int, str]]) -> str:
    if not files:
        raise ArtifactIntegrityError("immutable model inventory contains no model files")
    digest = hashlib.sha256()
    seen: set[str] = set()
    for relative, size, file_digest in sorted(files):
        if relative in seen:
            raise ArtifactIntegrityError(f"duplicate model inventory path: {relative}")
        seen.add(relative)
        normalized = file_digest.lower().removeprefix("sha256:")
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ArtifactIntegrityError(f"invalid SHA-256 for model inventory path: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(normalized))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_digest(snapshot: str | Path) -> str:
    """Hash a model tree deterministically."""
    root = Path(snapshot)
    if not root.is_dir():
        raise ArtifactIntegrityError("immutable model snapshot is not a directory")
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ArtifactIntegrityError("immutable model snapshots may not contain symlinks")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ArtifactIntegrityError("immutable model snapshot contains no model files")

    def inventory_entry(path: Path) -> tuple[str, int, str]:
        relative = path.relative_to(root).as_posix()
        return relative, path.stat().st_size, sha256_file(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(files))) as executor:
        inventory = list(executor.map(inventory_entry, files))
    return model_digest_from_inventory(inventory)


def verify_snapshot(snapshot: str | Path, expected_digest: str) -> str:
    observed = snapshot_digest(snapshot)
    if observed != expected_digest:
        raise ArtifactIntegrityError(
            f"immutable artifact digest mismatch: expected {expected_digest}, observed {observed}"
        )
    return observed
