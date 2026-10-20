"""The checks in order. `validate()` returns None for a legal submission or one sentence the
miner sees as `error`. Files are checked from the listing before download; the rest from
config.json, the safetensors headers and sizes after it."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from loguru import logger as log

from common.config import ChainSpec
from model_validation import families
from model_validation.checks import (
    check_architecture,
    check_config_keys,
    check_files,
    check_index,
    check_json,
    check_layout,
    check_quant,
    check_values,
    check_weight_bytes,
    layout_from_dir,
)
from model_validation.families import Family, weight_bytes


@lru_cache
def load(family: str) -> Family:
    """The rules for the family a miner declared at r2activate."""
    return families.load(family)


def validate(model_dir: Path, *, spec: ChainSpec, family: Family, quant: str) -> str | None:
    files = [p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()]
    if reason := check_files(files, spec) or check_json(model_dir) or check_index(model_dir):
        return reason
    config = json.loads((model_dir / "config.json").read_text())
    layout = layout_from_dir(model_dir)
    if reason := (
        check_config_keys(config, family.genesis_config)
        or check_architecture(config, family)
        or check_quant(config, quant)
        or check_layout(layout, family, config)
        or check_weight_bytes(layout, family)
        or check_values(model_dir)
    ):
        return reason
    log.info(
        "{} counted weight bytes: {}",
        model_dir.name,
        weight_bytes(layout, family.uncounted_prefixes),
    )
    return None
