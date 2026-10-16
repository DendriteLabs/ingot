"""Fingerprint genesis into the dedup bank, the model every submission is compared with, so a
requantized or disguised copy of it is caught. Once per bank and secret, on the GPU box:

    uv run --extra dedup scripts/seed_dedup_bank.py
"""

from __future__ import annotations

from common import models
from common.config import chain_spec, settings
from model_dedup.bank import BANK, GENESIS, Bank
from model_dedup.fingerprint import fingerprint_dir, load_secret


def main() -> None:
    s, spec = settings(), chain_spec()
    secret = load_secret(s)
    # connect (and create the index) before the ~10 min fingerprint, so a broken bank fails now
    bank = Bank.connect(s)
    uri = f"{spec.genesis_repo}@{spec.genesis_revision}"
    fp = fingerprint_dir(
        models.fetch(uri, s.model_cache_dir),
        teacher_dir=models.snapshot(spec.teacher_repo, spec.teacher_revision),
        secret=secret,
        device=s.dedup_device,
    )
    bank.put(GENESIS, fp, status=BANK, model_uri=uri)
    print(f"seeded {GENESIS} ({uri}): {len(fp['tensors'])} tensors, {fp['secs']}s")


if __name__ == "__main__":
    main()
