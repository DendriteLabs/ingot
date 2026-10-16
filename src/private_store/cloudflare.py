from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

API_ROOT = "https://api.cloudflare.com/client/v4"
R2_BUCKET_WRITE_PERMISSION = "Workers R2 Storage Bucket Item Write"


@dataclass(frozen=True, slots=True)
class ParentToken:
    token_id: str
    access_key_id: str
    secret_access_key: str


class CloudflareR2TokenGateway:
    """Small idempotent boundary around per-registration Cloudflare API tokens."""

    def __init__(
        self,
        http_client: Any,
        *,
        account_id: str,
        management_token: str,
        bucket: str,
        jurisdiction: str = "default",
    ) -> None:
        if not all((account_id, management_token, bucket, jurisdiction)):
            raise ValueError("Cloudflare token gateway configuration is incomplete")
        self.http = http_client
        self.account_id = account_id
        self.management_token = management_token
        self.bucket = bucket
        self.jurisdiction = jurisdiction
        self._permission_id: str | None = None

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(
            method,
            f"{API_ROOT}{path}",
            headers={"Authorization": f"Bearer {self.management_token}"},
            **kwargs,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            messages = [item.get("message", "unknown") for item in payload.get("errors", [])]
            raise RuntimeError(f"Cloudflare API rejected {method} {path}: {messages}")
        return payload.get("result")

    def _permission_group_id(self) -> str:
        if self._permission_id is None:
            groups = self._request("GET", f"/accounts/{self.account_id}/tokens/permission_groups")
            matches = [
                group["id"] for group in groups if group.get("name") == R2_BUCKET_WRITE_PERMISSION
            ]
            if len(matches) != 1:
                raise RuntimeError("could not uniquely resolve the R2 bucket-write permission")
            self._permission_id = matches[0]
        return self._permission_id

    def _existing_token_ids(self, token_name: str) -> list[str]:
        result = self._request("GET", f"/accounts/{self.account_id}/tokens")
        return [item["id"] for item in result if item.get("name") == token_name]

    def create_parent_token(self, token_name: str) -> ParentToken:
        # A token value is returned only once. An orphan left by a crash must be
        # revoked and recreated so the durable state can obtain its secret.
        for token_id in self._existing_token_ids(token_name):
            self.revoke_parent_token(token_id)
        resource = (
            f"com.cloudflare.edge.r2.bucket.{self.account_id}_{self.jurisdiction}_{self.bucket}"
        )
        result = self._request(
            "POST",
            f"/accounts/{self.account_id}/tokens",
            json={
                "name": token_name,
                "policies": [
                    {
                        "effect": "allow",
                        "permission_groups": [{"id": self._permission_group_id()}],
                        "resources": {resource: "*"},
                    }
                ],
            },
        )
        token_id = result.get("id")
        value = result.get("value")
        if not token_id or not value:
            raise RuntimeError("created Cloudflare token omitted its one-time value")
        return ParentToken(
            token_id=token_id,
            access_key_id=token_id,
            secret_access_key=hashlib.sha256(value.encode()).hexdigest(),
        )

    def revoke_parent_token(self, token_id: str) -> None:
        try:
            self._request("DELETE", f"/accounts/{self.account_id}/tokens/{token_id}")
        except Exception as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                return
            raise
