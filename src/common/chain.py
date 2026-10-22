"""Everything that touches the Bittensor chain: commitment scan, metagraph, swaps, weights.

`bittensor` is imported inside functions so the rest of the package imports fast and the
tests never need it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger as log

_BLOCK_HASH_CACHE: dict[int, str] = {}
_SIGNAL_PREFIXES = ("r2activate:", "r2ready:")


@dataclass(frozen=True)
class PrivateSignal:
    """An `r2activate:` or `r2ready:` commitment a miner revealed on chain."""

    kind: str  # "activate" | "ready"
    netuid: int
    block: int
    uid: int | None
    hotkey: str
    payload: str
    block_hash: str | None = None
    coldkey: str | None = None


@dataclass(frozen=True)
class SwapEvent:
    """A uid whose hotkey changed without re-registration: swap_hotkey, not churn."""

    uid: int
    old_hotkey: str
    new_hotkey: str
    registration_block: int

    def detail(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "old_hotkey": self.old_hotkey,
            "new_hotkey": self.new_hotkey,
            "registration_block": self.registration_block,
        }


def connect(network: str) -> Any:
    import bittensor as bt

    log.info("connecting to bittensor network={}", network)
    return bt.Subtensor(network=network)


def current_block(subtensor: Any) -> int:
    return int(subtensor.get_current_block())


def block_hash(subtensor: Any, block: int) -> str | None:
    if block not in _BLOCK_HASH_CACHE:
        try:
            _BLOCK_HASH_CACHE[block] = str(subtensor.get_block_hash(block))
        except Exception as exc:
            log.debug("get_block_hash({}) failed: {}", block, exc)
            return None
    return _BLOCK_HASH_CACHE[block]


def registration_burn_rao(subtensor: Any, netuid: int, block: int | None = None) -> int | None:
    """What registering a hotkey on `netuid` burns at `block`; None without state for that block.
    A normal node keeps a few hundred blocks of state, an archive node all of them."""
    cost = subtensor.recycle(netuid, block=block)
    return None if cost is None else int(cost.rao)


def block_time(subtensor: Any, block: int) -> datetime:
    """When `block` was produced, from its Timestamp.Now (milliseconds)."""
    millis = subtensor.substrate.query(
        "Timestamp", "Now", block_hash=block_hash(subtensor, block)
    ).value
    return datetime.fromtimestamp(millis / 1000, UTC)


def _decode_commitment(hotkey: str, data: Any) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for text, block in data:
        if not isinstance(text, str):
            raise ValueError(f"unexpected commitment payload type {type(text).__name__}")
        raw = bytes.fromhex(text[2:]) if text.startswith(("0x", "0X")) else text.encode("latin-1")
        if not raw:
            raise ValueError("empty commitment payload")
        mode = raw[0] & 0b11
        offset = 1 if mode == 0 else 2 if mode == 1 else 4
        out.append((int(block), raw[offset:].decode("utf-8", errors="ignore")))
    return out


def iter_revealed(
    subtensor: Any, netuid: int, block: int | None = None
) -> Iterator[tuple[str, int, str]]:
    """Every revealed commitment on the subnet as (hotkey, block, payload)."""
    qm = subtensor.query_map(
        module="Commitments", name="RevealedCommitments", params=[netuid], block=block
    )
    for k, v in qm:
        hotkey = str(getattr(k, "value", k))
        try:
            entries = _decode_commitment(hotkey, getattr(v, "value", v))
        except Exception as exc:
            log.debug("failed to decode revealed commitment for {}: {}", hotkey, exc)
            continue
        for block_number, payload in entries:
            yield hotkey, block_number, payload


def iter_committed(subtensor: Any, netuid: int) -> Iterator[tuple[str, int, str]]:
    """Every plain (not timelocked) commitment on the subnet as (hotkey, block, payload)."""
    for k, v in subtensor.query_map(module="Commitments", name="CommitmentOf", params=[netuid]):
        hotkey, value = str(getattr(k, "value", k)), getattr(v, "value", v)
        payload = ""
        for entry in value["info"]["fields"]:
            for kind, data in entry.items():
                if kind.startswith("Raw") and isinstance(data, str):
                    payload += bytes.fromhex(data.removeprefix("0x")).decode("utf-8", "ignore")
        yield hotkey, int(value["block"]), payload


def scan_signals(
    subtensor: Any,
    netuid: int,
    *,
    uids: dict[str, int],
    owners: dict[str, str],
    at_block: int | None = None,
    start_block: int = 0,
) -> list[PrivateSignal]:
    """Private-store signals revealed after `start_block`. Public commits are ignored."""
    signals: list[PrivateSignal] = []
    for hotkey, block, data in iter_revealed(subtensor, netuid, at_block):
        if block <= start_block or not data.startswith(_SIGNAL_PREFIXES):
            continue
        kind = "activate" if data.startswith("r2activate:") else "ready"
        # ready becomes a submission; its block hash seeds the eval sample draw
        bh = block_hash(subtensor, block) if kind == "ready" else None
        signals.append(
            PrivateSignal(
                kind, netuid, block, uids.get(hotkey), hotkey, data, bh, owners.get(hotkey)
            )
        )
    return signals


def metagraph_snapshot(
    subtensor: Any, netuid: int, block: int
) -> tuple[list[tuple[int, str, int]], dict[str, str]]:
    """(uid, hotkey, registration_block) for every neuron, plus hotkey → coldkey."""
    meta = subtensor.metagraph(netuid, block=block)
    reg_blocks: dict[int, int] = {}
    qm = subtensor.query_map(
        module="SubtensorModule", name="BlockAtRegistration", params=[netuid], block=block
    )
    for k, v in qm:
        reg_blocks[int(getattr(k, "value", k))] = int(getattr(v, "value", v))
    rows = [(int(n.uid), str(n.hotkey), reg_blocks.get(int(n.uid), 0)) for n in meta.neurons]
    owners = {str(n.hotkey): str(n.coldkey) for n in meta.neurons if n.coldkey}
    return rows, owners


def find_swaps(
    prior: dict[int, tuple[str, int | None]], current: list[tuple[int, str, int]]
) -> list[SwapEvent]:
    """uids whose hotkey changed while BlockAtRegistration did not."""
    swaps: list[SwapEvent] = []
    for uid, hotkey, reg_block in current:
        prev = prior.get(uid)
        if prev is None:
            continue
        prev_hotkey, prev_reg_block = prev
        if prev_hotkey != hotkey and prev_reg_block is not None and reg_block == prev_reg_block:
            swaps.append(SwapEvent(uid, prev_hotkey, hotkey, reg_block))
    return swaps


def hotkey_owner(subtensor: Any, hotkey: str, block: int | None = None) -> str | None:
    from scalecodec.utils.ss58 import ss58_encode

    owner = subtensor.query_subtensor("Owner", params=[hotkey], block=block)
    value = getattr(owner, "value", owner)
    if value is None or str(value) == ss58_encode(b"\x00" * 32, ss58_format=42):
        return None
    return str(value)


def confirm_swaps(subtensor: Any, swaps: list[SwapEvent], block: int) -> list[SwapEvent]:
    """Keep the swaps whose old hotkey no longer has an owner (churn keeps its owner)."""
    confirmed = []
    for swap in swaps:
        owner = hotkey_owner(subtensor, swap.old_hotkey, block)
        if owner is None:
            confirmed.append(swap)
        else:
            log.warning(
                "swap candidate rejected: uid {} old hotkey still owned by {}", swap.uid, owner
            )
    return confirmed


def build_wallet(coldkey: str, hotkey: str, path: str = "") -> Any:
    import bittensor as bt

    kwargs: dict[str, str] = {"name": coldkey, "hotkey": hotkey}
    if path:
        kwargs["path"] = path
    return bt.Wallet(**kwargs)


def reveal_commit(wallet: Any, *, netuid: int, network: str, payload: str) -> None:
    """Publish `payload` as a revealed commitment; raises when the chain rejects it."""
    result = connect(network).set_reveal_commitment(
        wallet=wallet, netuid=netuid, data=payload, blocks_until_reveal=1
    )
    if not getattr(result, "success", True):
        raise RuntimeError(f"commit failed: {getattr(result, 'message', result)}")
    log.info("committed: {}", payload)


class WeightClient:
    """set_weights with the validator's wallet, plus what the incentive rule needs."""

    def __init__(self, *, coldkey: str, hotkey: str, wallet_path: str, network: str) -> None:
        self.wallet = build_wallet(coldkey, hotkey, wallet_path)
        self.subtensor = connect(network)

    def block(self) -> int:
        return current_block(self.subtensor)

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]:
        return {int(n.uid): str(n.hotkey) for n in self.subtensor.metagraph(netuid).neurons}

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> tuple[bool, str]:
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            # without inclusion "success" only means submitted: an extrinsic the chain rejects
            # (rate limit, version key) would be recorded as set and not retried
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if isinstance(result, tuple):
            return bool(result[0]), str(result[1] or "") if len(result) > 1 else ""
        return bool(getattr(result, "success", result)), str(getattr(result, "message", "") or "")
