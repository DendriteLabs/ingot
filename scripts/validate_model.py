from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.config import ChainSpec, Quant, chain_spec
from model_validation import pipeline
from model_validation.checks import (
    check_architecture,
    check_config_keys,
    check_files,
    check_index,
    check_json,
    check_layout,
    check_quant,
    check_weight_bytes,
    layout_from_dir,
)
from model_validation.families import weight_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the model validation checks on a local model")
    spec = chain_spec()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--family", default=spec.family)
    parser.add_argument("--quant", default=spec.quant)
    args = parser.parse_args()
    model_dir = args.path
    pair = spec.pair(args.family, args.quant)

    print(f"upload limit {pair.upload_limit} bytes")
    if pair.format == "gguf":
        return _gguf(model_dir, spec, pair)
    family = pipeline.load(args.family)
    print(f"model   {model_dir}")
    print(f"family  {family.name} (genesis tensors {family.genesis_weight_bytes} bytes)")
    print()

    files = [
        rel
        for p in model_dir.rglob("*")
        if p.is_file() and not (rel := p.relative_to(model_dir).as_posix()).startswith(".cache/")
    ]
    config = json.loads((model_dir / "config.json").read_text())
    layout = layout_from_dir(model_dir)
    results = [
        ("files", check_files(files, spec, pair.format)),
        ("json", check_json(model_dir)),
        ("index", check_index(model_dir)),
        ("config keys", check_config_keys(config, family.genesis_config)),
        ("architecture", check_architecture(config, family)),
        ("quant", check_quant(config, args.quant)),
        ("layout", check_layout(layout, family, config)),
        ("weight bytes", check_weight_bytes(layout, family)),
    ]
    _print(results)

    print()
    print(f"tensors         {len(layout)}")
    size = weight_bytes(layout, family.uncounted_prefixes)
    print(
        f"counted weights {size} bytes ({size / family.genesis_weight_bytes:.2%} of all of genesis)"
    )
    failed = any(reason for _, reason in results)
    print(f"verdict         {'INVALID' if failed else 'VALID'}")
    return 1 if failed else 0


def _gguf(path: Path, spec: ChainSpec, pair: Quant) -> int:
    """A gguf quant: the one file (or the folder holding it) through every gguf check."""
    from model_validation.gguf_checks import report

    files = (
        [path.name]
        if path.is_file()
        else [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()]
    )
    print(f"model   {path}")
    print(f"quant   {pair.family}.{pair.name} (genesis file {pair.weight_limit} bytes)")
    print()
    results = [("files", check_files(files, spec, pair.format))]
    if results[0][1] is None:
        results += report(path, pair, pair.family)
    _print(results)
    failed = any(reason for _, reason in results)
    print()
    print(f"verdict         {'INVALID' if failed else 'VALID'}")
    return 1 if failed else 0


def _print(results: list[tuple[str, str | None]]) -> None:
    for name, reason in results:
        shown = reason if not reason or len(reason) <= 400 else reason[:400] + " ..."
        print(f"{'FAIL' if reason else 'PASS'}  {name:<13} {shown or ''}")


if __name__ == "__main__":
    sys.exit(main())
