from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from typing import Any

from botocore.exceptions import ClientError

from private_store.contracts import CANONICAL_MODEL_PREFIX, parse_model_prefix
from private_store.digests import ArtifactIntegrityError


class UploadQuotaExceeded(ArtifactIntegrityError):
    """A private registration prefix exceeds the model upload limit."""


class MailboxInvariantError(RuntimeError):
    pass


MAX_MINER_UPLOAD_OBJECTS = 4096  # a real model is <200 files; caps HEAD/list fan-out
MODEL_PREFIX_ROOTS = ("models/registrations/", "models/attempts/")


class MailboxStore:
    """Publish encrypted credential generations and remove them after revocation."""

    _KEY = re.compile(r"^mailbox/v1/[0-9a-f]{64}/generations/[0-9]{20}\.bin$")

    def __init__(self, s3_client: Any, *, bucket: str) -> None:
        if not bucket:
            raise ValueError("mailbox bucket is required")
        self.s3 = s3_client
        self.bucket = bucket

    @staticmethod
    def _digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def publish(self, key: str, ciphertext: bytes) -> None:
        try:
            existing = self.s3.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status != 404 and code not in {"NoSuchKey", "NotFound"}:
                raise
        else:
            body = existing["Body"]
            try:
                observed = body.read()
            finally:
                body.close()
            if self._digest(observed) != self._digest(ciphertext):
                raise MailboxInvariantError("mailbox generation already contains different bytes")
            return
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=ciphertext,
            ContentType="application/octet-stream",
            CacheControl="no-store, max-age=0",
            Metadata={"sha256": self._digest(ciphertext)},
        )

    def put_status(self, key: str, body: bytes) -> None:
        """Overwrite a plaintext status object.

        publish() is write-once by design (it raises when the bytes differ), and delete()
        only accepts credential-generation keys, so a status that changes on every
        transition needs its own plain put.
        """
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )

    def delete(self, keys: Iterable[str]) -> int:
        selected = sorted(set(keys))
        if any(not self._KEY.fullmatch(key) for key in selected):
            raise MailboxInvariantError("refusing to delete a non-mailbox object")
        for offset in range(0, len(selected), 1000):
            batch = selected[offset : offset + 1000]
            response = self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )
            if response.get("Errors"):
                raise RuntimeError("R2 failed to delete one or more mailbox credentials")
        return len(selected)


class R2UploadController:
    """Verify direct miner uploads in the private model bucket."""

    def __init__(
        self,
        s3_client: Any,
        *,
        private_model_bucket: str,
    ) -> None:
        if not private_model_bucket:
            raise ValueError("private model bucket is required")
        self.s3 = s3_client
        self.private_model_bucket = private_model_bucket

    def _multipart_uploads(self, prefix: str) -> tuple[dict[str, Any], ...]:
        uploads: list[dict[str, Any]] = []
        key_marker: str | None = None
        upload_marker: str | None = None
        while True:
            request: dict[str, Any] = {
                "Bucket": self.private_model_bucket,
                "Prefix": prefix,
            }
            if key_marker:
                request["KeyMarker"] = key_marker
            if upload_marker:
                request["UploadIdMarker"] = upload_marker
            response = self.s3.list_multipart_uploads(**request)
            uploads.extend(dict(item) for item in response.get("Uploads", ()))
            if not response.get("IsTruncated"):
                return tuple(uploads)
            key_marker = response.get("NextKeyMarker")
            upload_marker = response.get("NextUploadIdMarker")
            if not key_marker:
                raise ArtifactIntegrityError(
                    "truncated multipart listing omitted its continuation marker"
                )

    def quota_breach(
        self, model_prefix: str, max_bytes: int, allowed: Callable[[str], bool]
    ) -> str | None:
        """Cheap over-quota probe for the live-upload window (early-exit).

        Returns a reason string once the prefix crosses the byte or object cap,
        else None. Bounded cost: stops paging the moment a cap is exceeded, so
        a petabyte or millions-of-objects upload is caught in a few list calls.
        Parts of unfinished multipart uploads count too: R2 stores and bills them.
        A file name `allowed` refuses (a .py, a subfolder) is a breach as soon as it appears.
        """
        total = 0
        for upload in self._multipart_uploads(model_prefix):
            if not allowed(name := upload["Key"][len(model_prefix) :]):
                return f"file not allowed: {name}"
            total += self._part_bytes(upload["Key"], upload["UploadId"])
            if total > max_bytes:
                return (
                    f"{total} bytes (incl. unfinished multipart) exceeds the {max_bytes}-byte limit"
                )
        count = 0
        continuation: str | None = None
        while True:
            request: dict[str, Any] = {
                "Bucket": self.private_model_bucket,
                "Prefix": model_prefix,
            }
            if continuation:
                request["ContinuationToken"] = continuation
            response = self.s3.list_objects_v2(**request)
            for item in response.get("Contents", []):
                if not allowed(name := item["Key"][len(model_prefix) :]):
                    return f"file not allowed: {name}"
                count += 1
                total += int(item["Size"])
                if total > max_bytes:
                    return f"{total} bytes exceeds the {max_bytes}-byte limit"
                if count > MAX_MINER_UPLOAD_OBJECTS:
                    return f"more than {MAX_MINER_UPLOAD_OBJECTS} objects"
            if not response.get("IsTruncated"):
                return None
            continuation = response.get("NextContinuationToken")
            if not continuation:
                return None

    def _part_bytes(self, key: str, upload_id: str) -> int:
        total = 0
        marker: int | None = None
        while True:
            request: dict[str, Any] = {
                "Bucket": self.private_model_bucket,
                "Key": key,
                "UploadId": upload_id,
            }
            if marker is not None:
                request["PartNumberMarker"] = marker
            response = self.s3.list_parts(**request)
            total += sum(int(part["Size"]) for part in response.get("Parts", ()))
            if not response.get("IsTruncated"):
                return total
            marker = response.get("NextPartNumberMarker")
            if marker is None:
                return total

    def _list_objects(self, bucket: str, prefix: str) -> dict[str, dict[str, Any]]:
        objects: dict[str, dict[str, Any]] = {}
        continuation: str | None = None
        while True:
            request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation:
                request["ContinuationToken"] = continuation
            response = self.s3.list_objects_v2(**request)
            for item in response.get("Contents", []):
                key = item["Key"]
                if key in objects:
                    raise ArtifactIntegrityError(f"R2 listing repeated object key {key!r}")
                if not key.endswith("/"):
                    objects[key] = item
            if not response.get("IsTruncated"):
                return objects
            continuation = response.get("NextContinuationToken")
            if not continuation:
                raise ArtifactIntegrityError("truncated R2 listing omitted continuation token")

    def verify_upload(
        self,
        *,
        model_prefix: str,
        registration_id: str,
        max_upload_bytes: int,
        allowed: Callable[[str], bool],
    ) -> None:
        """Check the frozen upload from the listing: canonical prefix, allowed names, size.
        The bytes themselves are hashed against the committed model digest at download."""
        try:
            prefix_registration_id, _attempt = parse_model_prefix(model_prefix)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"registration model prefix is not canonical: {exc}"
            ) from exc
        if prefix_registration_id != registration_id:
            raise ArtifactIntegrityError("registration model prefix is not canonical")
        objects = self._list_objects(self.private_model_bucket, model_prefix)
        if not objects:
            raise ArtifactIntegrityError("private model prefix is empty")
        if len(objects) > MAX_MINER_UPLOAD_OBJECTS:
            raise UploadQuotaExceeded(
                f"private model upload has {len(objects)} objects; "
                f"limit is {MAX_MINER_UPLOAD_OBJECTS}"
            )
        total_uploaded_bytes = sum(int(item["Size"]) for item in objects.values())
        if total_uploaded_bytes > max_upload_bytes:
            raise UploadQuotaExceeded(
                f"private model upload is {total_uploaded_bytes} bytes; "
                f"limit is {max_upload_bytes} bytes"
            )
        if bad := sorted(
            k[len(model_prefix) :] for k in objects if not allowed(k[len(model_prefix) :])
        ):
            raise ArtifactIntegrityError(f"files not allowed: {', '.join(bad)}")

    def retained_uploads(self) -> dict[str, Any]:
        """Map every prefix that still holds objects to its newest LastModified.

        The bucket, not the database, is the authority on what occupies storage: one
        registration can own several attempt prefixes, and a prefix orphaned by a
        crashed flow has no row at all. The newest object in a prefix is also when
        that upload finished, so no separate age column is needed.
        """
        newest: dict[str, Any] = {}
        for root in MODEL_PREFIX_ROOTS:
            nested = root.endswith("attempts/")
            for key, item in self._list_objects(self.private_model_bucket, root).items():
                registration_id, _, rest = key[len(root) :].partition("/")
                attempt, _, tail = rest.partition("/")
                if not rest or (nested and not tail):
                    continue
                prefix = f"{root}{registration_id}/" + (f"{attempt}/" if nested else "")
                if not CANONICAL_MODEL_PREFIX.fullmatch(prefix):
                    continue
                modified = item.get("LastModified")
                if modified is None:
                    continue
                current = newest.get(prefix)
                newest[prefix] = modified if current is None else max(current, modified)
        return newest

    def abort_multipart_uploads(self, model_prefix: str) -> int:
        if not CANONICAL_MODEL_PREFIX.fullmatch(model_prefix):
            raise ValueError(f"refusing to bulk-abort a non-registration prefix: {model_prefix!r}")
        aborted = 0
        for upload in self._multipart_uploads(model_prefix):
            self.s3.abort_multipart_upload(
                Bucket=self.private_model_bucket,
                Key=upload["Key"],
                UploadId=upload["UploadId"],
            )
            aborted += 1
        return aborted

    def cleanup_model_prefix(self, model_prefix: str) -> int:
        if not CANONICAL_MODEL_PREFIX.fullmatch(model_prefix):
            raise ValueError(f"refusing to bulk-delete a non-registration prefix: {model_prefix!r}")
        # Abort dangling multipart uploads too — otherwise a prefix whose only content is an
        # unfinished multipart (e.g. a verify-FAILED submission) leaks R2 storage forever.
        aborted = self.abort_multipart_uploads(model_prefix)
        objects = self._list_objects(self.private_model_bucket, model_prefix)
        if not objects:
            return aborted
        keys = sorted(objects)
        for offset in range(0, len(keys), 1000):
            self.s3.delete_objects(
                Bucket=self.private_model_bucket,
                Delete={
                    "Objects": [{"Key": key} for key in keys[offset : offset + 1000]],
                    "Quiet": True,
                },
            )
        return len(keys) + aborted
