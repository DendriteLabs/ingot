from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.config import chain_spec
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
    weight_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the model validation checks on a local model")
    spec = chain_spec()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--family", default=spec.family)
    parser.add_argument("--quant", default=spec.quant)
    args = parser.parse_args()
    model_dir = args.path

    print(f"upload limit {spec.upload_limit(args.family, args.quant)} bytes")
    family = pipeline.load(args.family)
    print(f"model   {model_dir}")
    print(f"family  {family.name} (genesis safetensors {family.genesis_weight_bytes} bytes)")
    print()

    files = [
        rel
        for p in model_dir.rglob("*")
        if p.is_file() and not (rel := p.relative_to(model_dir).as_posix()).startswith(".cache/")
    ]
    config = json.loads((model_dir / "config.json").read_text())
    results = [
        ("files", check_files(files, spec)),
        ("json", check_json(model_dir)),
        ("index", check_index(model_dir)),
        ("config keys", check_config_keys(config, family.genesis_config)),
        ("architecture", check_architecture(config, family)),
        ("quant", check_quant(config, args.quant)),
        ("layout", check_layout(layout_from_dir(model_dir), family, config)),
        ("weight bytes", check_weight_bytes(model_dir, family)),
    ]
    for name, reason in results:
        shown = reason if not reason or len(reason) <= 400 else reason[:400] + " ..."
        print(f"{'FAIL' if reason else 'PASS'}  {name:<13} {shown or ''}")

    print()
    layout = layout_from_dir(model_dir)
    print(f"tensors         {len(layout)}")
    size = weight_bytes(model_dir)
    print(f"safetensors     {size} bytes ({size / family.genesis_weight_bytes:.2%} of genesis)")
    failed = any(reason for _, reason in results)
    print(f"verdict         {'INVALID' if failed else 'VALID'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
