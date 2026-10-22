"""The OpenSearch bank (docker compose up -d): a miner's own models are never compared with
their next one, audited fingerprints are kept but never compared, genesis always is."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from common.config import Settings
from model_dedup import config
from model_dedup.bank import AUDIT, BANK, GENESIS, Bank
from tests.conftest import fingerprint, quantization_error

pytest.importorskip("opensearchpy")


@pytest.fixture
def bank() -> Iterator[Bank]:
    s = Settings(
        _env_file=None,
        opensearch_host=os.environ.get("INGOT_OPENSEARCH_HOST", "127.0.0.1"),
        opensearch_port=int(os.environ.get("INGOT_OPENSEARCH_PORT", "9270")),
        opensearch_index=f"ingot_test_{uuid.uuid4().hex[:8]}",
    )
    try:
        bank = Bank.connect(s, "q2")
    except Exception as exc:
        pytest.skip(f"no OpenSearch at {s.opensearch_url} ({exc}); run `docker compose up -d`")
    bank.put(GENESIS, fingerprint(quantization_error(0)), status=BANK, model_uri="org/genesis")
    yield bank
    bank.client.indices.delete(index=bank.index)


def test_a_miners_own_models_are_never_compared(bank):
    error = quantization_error(1)
    bank.put("old", fingerprint(error), status=BANK, model_uri="m1", hotkey="hk-old", coldkey="ck")
    bank.put(
        "other", fingerprint(quantization_error(2)), status=BANK, model_uri="m2", hotkey="hk-x"
    )

    assert set(bank.shortlist(fingerprint(error), hotkey="hk-new", coldkey="ck")) == {
        GENESIS,
        "other",
    }, "same coldkey, new hotkey"
    assert "old" not in bank.shortlist(fingerprint(error), hotkey="hk-old", coldkey=None)
    assert "old" in bank.shortlist(fingerprint(error), hotkey="hk-y", coldkey="ck-y")


def test_audited_fingerprints_are_kept_but_never_compared(bank):
    error = quantization_error(3)
    bank.put("copy", fingerprint(error), status=AUDIT, model_uri="m3", hotkey="hk-c")
    assert set(bank.shortlist(fingerprint(error), hotkey="hk-z", coldkey=None)) == {GENESIS}
    assert bank.client.exists(index=bank.index, id="copy")


def test_identical_tensors_are_found_beyond_the_nearest(bank, monkeypatch):
    monkeypatch.setattr(config, "NEAREST_K", 1)
    near = quantization_error(4)
    bank.put("near", fingerprint(near), status=BANK, model_uri="m4", hotkey="hk-n")
    bank.put(
        "twin",
        fingerprint(quantization_error(5), tensors_hash="same-bytes"),
        status=BANK,
        model_uri="m5",
        hotkey="hk-t",
    )
    probe = fingerprint(near, tensors_hash="same-bytes")
    assert set(bank.shortlist(probe, hotkey="hk-p", coldkey=None)) == {GENESIS, "near", "twin"}


def test_the_seeded_genesis_names_its_secret(bank):
    assert bank.genesis_scope() == ("test", "test")
