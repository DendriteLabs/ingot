"""`ingot`: the miner CLI. Submissions go through the private R2 store only."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from miner import env

env.load()

_NETUID = int(os.environ.get("INGOT_NETUID", "97"))
_NETWORK = os.environ.get("INGOT_NETWORK", "finney")
_COLDKEY = os.environ.get("INGOT_COLDKEY")
_HOTKEY = os.environ.get("INGOT_HOTKEY")


def _wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coldkey", default=_COLDKEY, required=_COLDKEY is None)
    parser.add_argument("--hotkey", default=_HOTKEY, required=_HOTKEY is None)
    parser.add_argument("--netuid", type=int, default=_NETUID)
    parser.add_argument("--network", default=_NETWORK)
    parser.add_argument("--yes", action="store_true", help="skip the confirm prompts")


def _pair_args(parser: argparse.ArgumentParser) -> None:
    from common.config import chain_spec

    spec = chain_spec()
    parser.add_argument("--family", default=spec.family, help="model family you quantized")
    parser.add_argument("--quant", default=spec.quant, help="quantization you declare, e.g. fp8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ingot", description="Ingot miner CLI")
    sub = p.add_subparsers(dest="cmd")

    _wallet_args(sub.add_parser("register", help="burned-register the hotkey on the subnet"))

    check = sub.add_parser("check-model", help="run the validator's checks on a local model")
    check.add_argument("--path", required=True)
    _pair_args(check)

    submit = sub.add_parser("submit-private", help="activate → upload → ready, end to end")
    submit.add_argument("--path", required=True)
    submit.add_argument("--skip-check", action="store_true")
    _pair_args(submit)
    _wallet_args(submit)

    act = sub.add_parser("activate", help="commit r2activate to request credentials")
    _pair_args(act)
    _wallet_args(act)

    up = sub.add_parser("upload-private", help="poll credentials, upload the model files")
    up.add_argument("--path", required=True)
    up.add_argument("--skip-check", action="store_true")
    _pair_args(up)
    _wallet_args(up)

    rd = sub.add_parser("ready", help="commit r2ready to freeze the upload and start verification")
    rd.add_argument("--model-digest", required=True)
    _wallet_args(rd)
    return p


def check_model(path: str, family: str, quant: str) -> str | None:
    """None when the model would pass validation, else the reason it would not."""
    from common.config import chain_spec
    from model_validation import pipeline

    spec = chain_spec()
    try:
        limit = spec.upload_limit(family, quant)
    except ValueError as exc:
        return str(exc)
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    if size > limit:
        return f"{size} bytes exceeds the {limit}-byte upload limit for {family}.{quant}"
    return pipeline.validate(Path(path), spec=spec, family=pipeline.load(family), quant=quant)


def _require_valid(path: str, *, family: str, quant: str, skip: bool) -> bool:
    if skip:
        print("model check skipped (--skip-check)")
        return True
    reason = check_model(path, family, quant)
    print(f"INVALID — {reason}" if reason else "VALID")
    return reason is None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.cmd is None:
        parser.print_help()
        return 0
    if args.cmd == "check-model":
        return (
            0 if _require_valid(args.path, family=args.family, quant=args.quant, skip=False) else 1
        )

    from miner import submit, wallet

    if args.cmd == "register":
        uid = wallet.register(args.coldkey, args.hotkey, args.netuid, args.network, yes=args.yes)
        return 0 if uid is not None else 1

    w = wallet.build_wallet(args.coldkey, args.hotkey)
    if args.cmd == "submit-private":
        if not _require_valid(
            args.path, family=args.family, quant=args.quant, skip=args.skip_check
        ):
            return 1
        submit.submit_private(
            w,
            netuid=args.netuid,
            network=args.network,
            family=args.family,
            quant=args.quant,
            local_dir=args.path,
            yes=args.yes,
        )
    elif args.cmd == "activate":
        rid = submit.activate(
            w,
            netuid=args.netuid,
            network=args.network,
            family=args.family,
            quant=args.quant,
            yes=args.yes,
        )
        print("registration_id:", rid)
    elif args.cmd == "upload-private":
        if not _require_valid(
            args.path, family=args.family, quant=args.quant, skip=args.skip_check
        ):
            return 1
        ss58 = w.hotkey.ss58_address
        key = submit.submission_key(ss58)
        rid = submit.registration_id_for(ss58, netuid=args.netuid)
        submit.fetch_credentials(hotkey_ss58=ss58, rid=rid, signing_key=key)
        digest = submit.upload(hotkey_ss58=ss58, local_dir=args.path)
        print("model_digest:", digest)
        print("next: ingot ready --model-digest", digest)
    elif args.cmd == "ready":
        submit.ready(
            w,
            netuid=args.netuid,
            network=args.network,
            model_digest=args.model_digest,
            yes=args.yes,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
