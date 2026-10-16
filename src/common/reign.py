"""Kings: who gets paid and what they are called. The newest SLOTS kings each earn SHARE for
REIGN_DAYS after their coronation; whatever is left goes to BURN_UID. A king whose uid now
belongs to a different hotkey is treated as gone. Genesis (no uid) is never paid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

SLOTS = 5
REIGN_DAYS = 5.0
SHARE = 0.2
BURN_UID = 0

_ROMAN = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
    (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
)  # fmt: skip


@dataclass(frozen=True)
class King:
    uid: int | None
    hotkey: str | None
    crowned_at: datetime


def name(ordinal: int | None, suffix: str) -> str:
    """Ingot-I-<suffix> for the first crowned king, with the suffix of its family and quant
    (27B-FP8 today); genesis (no ordinal) is GENESIS."""
    if ordinal is None:
        return "GENESIS"
    out, n = "", ordinal
    for value, symbol in _ROMAN:
        count, n = divmod(n, value)
        out += symbol * count
    return f"Ingot-{out}-{suffix}"


def ends_at(crowned_at: datetime, *, reign_days: float = REIGN_DAYS) -> datetime:
    return crowned_at + timedelta(days=reign_days)


def paid(
    kings: Sequence[King],
    now: datetime,
    *,
    slots: int = SLOTS,
    reign_days: float = REIGN_DAYS,
) -> list[King]:
    newest = sorted(kings, key=lambda k: k.crowned_at, reverse=True)[:slots]
    return [
        k
        for k in newest
        if k.uid is not None and now <= ends_at(k.crowned_at, reign_days=reign_days)
    ]


def weights(
    kings: Sequence[King],
    now: datetime,
    uid_hotkeys: dict[int, str],
    *,
    slots: int = SLOTS,
    reign_days: float = REIGN_DAYS,
    share: float = SHARE,
    burn_uid: int = BURN_UID,
) -> dict[int, float]:
    """uid → weight, summing to 1.0."""
    if share * slots > 1 + 1e-9:
        raise ValueError(f"{slots} kings × {share} exceeds the whole emission")
    out: dict[int, float] = {}
    for king in paid(kings, now, slots=slots, reign_days=reign_days):
        if king.uid is not None and uid_hotkeys.get(king.uid) == king.hotkey:
            out[king.uid] = out.get(king.uid, 0.0) + share
    rest = round(1.0 - sum(out.values()), 9)
    if rest > 0:
        out[burn_uid] = out.get(burn_uid, 0.0) + rest
    return out
