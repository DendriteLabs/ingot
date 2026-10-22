"""Fingerprint a quant's genesis into its dedup bank, the model every submission of that quant is
compared with, so a requantized or disguised copy of it is caught. Once per bank and secret, on
the GPU box (a quant with dedup = "hash" banks genesis's file hash instead, no GPU):

    uv run --extra dedup scripts/seed_dedup_bank.py [--quant q2]
"""

from __future__ import annotations

import argparse

from common import models
from common.config import chain_spec, settings
from model_dedup.bank import BANK, GENESIS, Bank
from model_dedup.files import file_fingerprint


def main() -> None:
    s, spec = settings(), chain_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", default=spec.quant)
    quant = spec.pair(spec.family, parser.parse_args().quant)
    # connect (and create the index) before the ~10 min fingerprint, so a broken bank fails now
    bank = Bank.connect(s, quant.name, quant.dedup)
    uri = models.genesis_uri(quant)
    model_dir = models.fetch(uri, s.model_cache_dir)
    if quant.dedup == "hash":
        fp = file_fingerprint(model_dir)
    else:
        from model_dedup.fingerprint import fingerprint_dir, load_secret  # torch

        teacher = models.snapshot(spec.teacher_repo, spec.teacher_revision)
        fp = fingerprint_dir(
            model_dir, teacher_dir=teacher, secret=load_secret(s), device=s.dedup_device
        )
    bank.put(GENESIS, fp, status=BANK, model_uri=uri)
    print(f"seeded {GENESIS} ({uri}) into {bank.index} ({quant.dedup} dedup)")


if __name__ == "__main__":
    main()
