from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from nacl import bindings
from nacl.exceptions import BadSignatureError
from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.signing import SigningKey, VerifyKey

_BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {char: index for index, char in enumerate(_BASE58_ALPHABET)}
_SS58_PREFIX = b"SS58PRE"


def _base58_encode(value: bytes) -> str:
    number = int.from_bytes(value, "big")
    encoded = ""
    while number:
        number, remainder = divmod(number, 58)
        encoded = _BASE58_ALPHABET[remainder] + encoded
    zeros = len(value) - len(value.lstrip(b"\0"))
    return "1" * zeros + (encoded or "")


def _base58_decode(value: str) -> bytes:
    number = 0
    for char in value:
        try:
            digit = _BASE58_INDEX[char]
        except KeyError as exc:
            raise ValueError("invalid base58 character in SS58 address") from exc
        number = number * 58 + digit
    decoded = number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b""
    zeros = len(value) - len(value.lstrip("1"))
    return b"\0" * zeros + decoded


def encode_ss58_public_key(public_key: bytes, *, network: int = 42) -> str:
    if len(public_key) != 32:
        raise ValueError("Ed25519 public key must contain 32 bytes")
    if not 0 <= network < 64:
        raise ValueError("only one-byte SS58 network prefixes are supported")
    payload = bytes([network]) + public_key
    checksum = hashlib.blake2b(_SS58_PREFIX + payload, digest_size=64).digest()[:2]
    return _base58_encode(payload + checksum)


def decode_ss58_public_key(address: str, *, expected_network: int = 42) -> bytes:
    decoded = _base58_decode(address)
    if len(decoded) != 35:
        raise ValueError("SS58 Ed25519 address must decode to 35 bytes")
    payload, checksum = decoded[:-2], decoded[-2:]
    if payload[0] != expected_network:
        raise ValueError("SS58 address uses an unexpected network prefix")
    expected = hashlib.blake2b(_SS58_PREFIX + payload, digest_size=64).digest()[:2]
    if checksum != expected:
        raise ValueError("SS58 checksum is invalid")
    return payload[1:]


def _signature_bytes(signature: str) -> bytes:
    if signature.startswith("0x"):
        try:
            raw = bytes.fromhex(signature[2:])
        except ValueError as exc:
            raise ValueError("signature is not valid hexadecimal") from exc
    else:
        try:
            raw = base64.b64decode(signature, validate=True)
        except Exception as exc:
            raise ValueError("signature is not valid base64") from exc
    if len(raw) != 64:
        raise ValueError("Ed25519 signature must contain 64 bytes")
    return raw


def encode_signature(signature: bytes) -> str:
    if len(signature) != 64:
        raise ValueError("Ed25519 signature must contain 64 bytes")
    return base64.b64encode(signature).decode()


def verify_hotkey_signature(hotkey: str, message: bytes, signature: str) -> None:
    try:
        VerifyKey(decode_ss58_public_key(hotkey)).verify(message, _signature_bytes(signature))
    except BadSignatureError as exc:
        raise ValueError("Ed25519 signature verification failed") from exc


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class MailboxCipher:
    """Sign credential envelopes and sealed-box encrypt them to the submission key."""

    def __init__(self, validator_signing_key: SigningKey, *, network: int = 42) -> None:
        self._signing_key = validator_signing_key
        self.network = network
        self.validator_identity = encode_ss58_public_key(
            bytes(validator_signing_key.verify_key), network=network
        )

    def build_envelope(
        self,
        *,
        netuid: int,
        hotkey: str,
        registration_id: str,
        generation: int,
        endpoint: str,
        private_model_bucket: str,
        allowed_prefix: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: str,
        expires_at: datetime,
        chain_generation: str,
    ) -> dict[str, Any]:
        unsigned: dict[str, Any] = {
            "protocol_version": 1,
            "validator_identity": self.validator_identity,
            "netuid": netuid,
            "hotkey": hotkey,
            "registration_id": registration_id,
            "credential_generation": generation,
            "r2_endpoint": endpoint,
            "private_model_bucket": private_model_bucket,
            "allowed_prefix": allowed_prefix,
            "credential_scope": "object-read-write",
            "revocation_event": "finalized_ready_signal",
            "submission_policy": "one_per_hotkey",
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "session_token": session_token,
            "expires_at": _utc_text(expires_at),
            "chain_generation": chain_generation,
            "signature_scheme": "ed25519",
        }
        signature = self._signing_key.sign(_canonical_json(unsigned)).signature
        return {**unsigned, "validator_signature": encode_signature(signature)}

    def encrypt_for_pubkey(self, envelope: Mapping[str, Any], ed25519_public: bytes) -> bytes:
        curve_public = bindings.crypto_sign_ed25519_pk_to_curve25519(ed25519_public)
        return bytes(SealedBox(PublicKey(curve_public)).encrypt(_canonical_json(envelope)))

    def create_ciphertext(
        self, *, submission_pubkey: bytes, **envelope_fields: Any
    ) -> tuple[bytes, dict]:
        envelope = self.build_envelope(**envelope_fields)
        return self.encrypt_for_pubkey(envelope, submission_pubkey), envelope

    @staticmethod
    def decrypt_for_miner(ciphertext: bytes, miner_signing_key: SigningKey) -> dict[str, Any]:
        curve_secret = bindings.crypto_sign_ed25519_sk_to_curve25519(
            bytes(miner_signing_key._signing_key)
        )
        plaintext = SealedBox(PrivateKey(curve_secret)).decrypt(ciphertext)
        envelope = json.loads(plaintext)
        signature = envelope.pop("validator_signature")
        verify_hotkey_signature(
            envelope["validator_identity"], _canonical_json(envelope), signature
        )
        return {**envelope, "validator_signature": signature}
