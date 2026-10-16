"""config.json may only hold keys genesis has (no auto_map, no custom code or remote
references), must reproduce every genesis value outside the family's free keys, and must say
it is the quantization the miner declared. The model must compute what genesis computes; the
weights, the block size and the bf16/fp8 split are the miner's."""

from __future__ import annotations

from typing import Any

from model_validation.families import Family, lookup


def check_architecture(config: dict[str, Any], family: Family) -> str | None:
    for key in family.arch_keys:
        got, want = lookup(config, key), family.arch_value(key)
        if got != want:
            return f"config.json {key} is {_short(got)}, genesis has {_short(want)}"
    return None


def check_config_keys(config: Any, genesis: Any, path: str = "") -> str | None:
    if not isinstance(config, dict):
        return "config.json must be an object" if not path else None
    for key, value in config.items():
        dotted = f"{path}{key}"
        if not isinstance(genesis, dict) or key not in genesis:
            return f"config.json has a key genesis does not: {dotted}"
        if isinstance(genesis[key], dict) and not isinstance(value, dict | None):
            return f"config.json {dotted} must be an object"
        if isinstance(value, dict) and (
            reason := check_config_keys(value, genesis[key], dotted + ".")
        ):
            return reason
    return None


def check_quant(config: dict[str, Any], quant: str) -> str | None:
    got = lookup(config, "quantization_config.quant_method")
    if got != quant:
        return f"config.json quantization_config.quant_method is {_short(got)}, declared {quant!r}"
    return None


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."
