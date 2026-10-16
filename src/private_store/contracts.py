from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass

PROTOCOL_VERSION = 1
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")
_ACTIVATION = re.compile(
    r"^r2activate:v2:(?P<family>[a-z0-9_]{1,32}):(?P<quant>[a-z0-9_]{1,32}):"
    r"(?P<pubkey>[A-Za-z0-9_-]{43})$"
)
_READY_SIGNAL = re.compile(r"^r2ready:v2:(?P<model_digest>[0-9a-f]{64})$")
CANONICAL_MODEL_PREFIX = re.compile(
    r"^models/(?:registrations/(?P<first>[0-9a-f]{64})"
    r"|attempts/(?P<retry>[0-9a-f]{64})/a(?P<attempt>[2-9]|[1-9][0-9]+))/$"
)


def _require_hash(value: str, field: str) -> str:
    if not _HEX_SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_ss58(value: str, field: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field} is not a canonical SS58-like identifier")
    return value


def _require_chain_generation(value: str) -> str:
    if not value or "|" in value or len(value) > 128:
        raise ValueError("chain_generation must be non-empty and delimiter-safe")
    return value


def registration_id(*, netuid: int, hotkey: str, chain_generation: str) -> str:
    """Derive one registration identity from public chain data.

    Hotkeys are one-shot on ingot, so the hotkey alone identifies the
    occupancy — no uid or registration block needed.
    """
    if netuid < 0:
        raise ValueError("netuid must be non-negative")
    _require_ss58(hotkey, "hotkey")
    _require_chain_generation(chain_generation)
    body = json.dumps(
        {"chain_generation": chain_generation, "hotkey": hotkey, "netuid": netuid},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"ingot-registration-v1\0" + body).hexdigest()


def model_prefix(registration_id: str, attempt: int = 1) -> str:
    """Return the R2 key prefix owning one upload attempt.

    Attempt 1 lives at `models/registrations/<rid>/` and retries under a sibling
    root, so no attempt's prefix contains another's: clearing one never touches the
    others, and each attempt's objects are inventoried on their own.
    """
    _require_hash(registration_id, "registration_id")
    if attempt < 1:
        raise ValueError("upload attempt must start at one")
    if attempt == 1:
        return f"models/registrations/{registration_id}/"
    return f"models/attempts/{registration_id}/a{attempt}/"


def parse_model_prefix(prefix: str) -> tuple[str, int]:
    """Recover (registration_id, attempt) from a prefix; the inverse of model_prefix."""
    match = CANONICAL_MODEL_PREFIX.fullmatch(prefix)
    if match is None:
        raise ValueError(f"not a canonical model prefix: {prefix!r}")
    attempt = match["attempt"]
    return (match["first"] or match["retry"]), int(attempt) if attempt else 1


@dataclass(frozen=True)
class Activation:
    family: str
    quant: str
    pubkey: bytes


def activation_signal_payload(submission_pubkey: bytes, *, family: str, quant: str) -> str:
    if len(submission_pubkey) != 32:
        raise ValueError("submission public key must contain 32 bytes (Ed25519)")
    encoded = base64.urlsafe_b64encode(submission_pubkey).rstrip(b"=").decode("ascii")
    payload = f"r2activate:v2:{family}:{quant}:{encoded}"
    parse_activation(payload)
    return payload


def parse_activation(payload: str) -> Activation:
    """The declared family, quantization and one-time Ed25519 key of an r2activate:v2 commit."""
    match = _ACTIVATION.fullmatch(payload)
    if match is None:
        raise ValueError(
            "activation signal is not a valid r2activate:v2:<family>:<quant>:<pubkey> commitment"
        )
    pubkey = base64.urlsafe_b64decode(match.group("pubkey") + "=")
    if len(pubkey) != 32:
        raise ValueError("activation signal must carry a 32-byte Ed25519 public key")
    return Activation(family=match["family"], quant=match["quant"], pubkey=pubkey)


def ready_signal_payload(model_digest: str) -> str:
    _require_hash(model_digest, "model_digest")
    return f"r2ready:v2:{model_digest}"


def parse_ready_signal(payload: str) -> str:
    """Return the model digest (digests.model_digest_from_inventory) from an r2ready:v2 commit."""
    match = _READY_SIGNAL.fullmatch(payload)
    if match is None:
        raise ValueError("ready signal is not a valid r2ready:v2:<model digest> commitment")
    return match.group("model_digest")


def mailbox_status_key(registration_id: str) -> str:
    """Return the public mailbox key carrying this registration's plaintext status.

    Deliberately outside CANONICAL mailbox key shape, so MailboxStore.delete refuses it and
    revoke/window-close can never wipe the record the miner reads.
    """
    _require_hash(registration_id, "registration_id")
    return f"mailbox/v1/{registration_id}/status.json"


def mailbox_object_key(registration_id: str, generation: int) -> str:
    """Return the immutable public mailbox key for one credential generation."""
    _require_hash(registration_id, "registration_id")
    if generation < 1:
        raise ValueError("credential generation must start at one")
    return f"mailbox/v1/{registration_id}/generations/{generation:020d}.bin"
