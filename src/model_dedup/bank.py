"""The fingerprint bank in OpenSearch. A submission is compared with the banked models of other
miners nearest to it, then its own fingerprint is stored: `bank` when unique, `audit` when not
(kept for review, never compared against)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from common.config import Settings
from model_dedup import config
from model_dedup.verdict import Verdict, sketches

GENESIS = "genesis"
BANK, AUDIT = "bank", "audit"
DIM = 1024
_MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "model_uri": {"type": "keyword"},
            "hotkey": {"type": "keyword"},
            "coldkey": {"type": "keyword"},
            "status": {"type": "keyword"},
            "version": {"type": "keyword"},
            "key_id": {"type": "keyword"},
            "tensors_hash": {"type": "keyword"},
            "vec": {"type": "knn_vector", "dimension": DIM},
            "verdict": {"type": "object", "enabled": False},
            "fingerprint": {"type": "object", "enabled": False},
        }
    },
}


def summary_vector(fp: dict) -> np.ndarray:
    """Unit vector whose dot products estimate the correlation of two fingerprints' errors.

    A count sketch of all tensor sketches: each entry goes to one of DIM buckets with a random
    sign, fixed per tensor name. Error ~1/sqrt(DIM) = 0.03 against >= 0.36 for every copy
    measured on Qwen3.8 and ~0 for honest pairs. No secret needed: it only mixes numbers that
    already are secret projections.
    """
    v = np.zeros(DIM)
    for name, s in sorted(sketches(fp).items()):
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8]))
        np.add.at(v, rng.integers(0, DIM, s.size), rng.choice((-1.0, 1.0), s.size) * s.ravel())
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def index_name(s: Settings, quant: str, mode: str = "full") -> str:
    """One index per quant and dedup mode: hash fingerprints never meet full ones."""
    return f"{s.opensearch_index}-{quant}" + ("-hash" if mode == "hash" else "")


class Bank:
    def __init__(self, client: Any, index: str, mode: str = "full") -> None:
        self.client, self.index, self.mode = client, index, mode
        if not client.indices.exists(index=index):
            client.indices.create(index=index, body=_MAPPING)

    @classmethod
    def connect(cls, s: Settings, quant: str, mode: str = "full") -> Bank:
        """The bank of one quant: fingerprints of different quants have different teachers and
        layouts and are never compared."""
        from opensearchpy import OpenSearch

        client = OpenSearch(hosts=[s.opensearch_url], timeout=60)
        return cls(client, index_name(s, quant, mode), mode)

    def genesis_scope(self) -> tuple[str, str] | None:
        """(version, key_id) of the seeded genesis fingerprint, or None before seeding."""
        if not self.client.exists(index=self.index, id=GENESIS):
            return None
        doc = self.client.get(index=self.index, id=GENESIS, _source_includes=["version", "key_id"])
        return doc["_source"]["version"], doc["_source"]["key_id"]

    def put(
        self,
        bank_id: str,
        fp: dict,
        *,
        status: str,
        model_uri: str,
        hotkey: str | None = None,
        coldkey: str | None = None,
        verdict: Verdict | None = None,
    ) -> None:
        body = {
            "model_uri": model_uri,
            "hotkey": hotkey,
            "coldkey": coldkey,
            "status": status,
            "version": fp["version"],
            "key_id": fp["key_id"],
            "tensors_hash": fp["tensors_hash"],
            "verdict": asdict(verdict) if verdict else None,
            "fingerprint": fp,
        }
        if self.mode == "full":
            body["vec"] = summary_vector(fp).tolist()
        # the next submission's shortlist must already see this one
        self.client.index(index=self.index, id=bank_id, body=body, refresh="wait_for")

    def shortlist(self, fp: dict, *, hotkey: str, coldkey: str | None) -> dict[str, dict]:
        """Banked fingerprints to compare in full: other miners' models with identical tensors,
        the NEAREST_K nearest, and genesis. The miner's own hotkey and coldkey are left out, so
        an earlier model of theirs never counts against the next."""
        scope = [
            {"term": {"status": BANK}},
            {"term": {"version": fp["version"]}},
            {"term": {"key_id": fp["key_id"]}},
        ]
        own = [{"term": {"hotkey": hotkey}}] + ([{"term": {"coldkey": coldkey}}] if coldkey else [])
        exact = {"bool": {"filter": [*scope, {"term": {"tensors_hash": fp["tensors_hash"]}}]}}
        exact["bool"]["must_not"] = own
        queries: list[tuple[dict, int]] = [(exact, 100)]
        # the hash mode compares identical files only: its fingerprint has no tensors to vectorize
        if self.mode == "full":
            nearest = {
                "script_score": {
                    "query": {"bool": {"filter": scope, "must_not": own}},
                    "script": {
                        "source": "knn_score",
                        "lang": "knn",
                        # unit vectors: L2 ranks like cosine
                        "params": {
                            "field": "vec",
                            "query_value": summary_vector(fp).tolist(),
                            "space_type": "l2",
                        },
                    },
                }
            }
            queries.append((nearest, config.NEAREST_K))
        ids = {GENESIS}
        for query, size in queries:
            body = {"size": size, "_source": False, "query": query}
            ids |= {
                h["_id"] for h in self.client.search(index=self.index, body=body)["hits"]["hits"]
            }
        docs = self.client.mget(
            index=self.index, body={"ids": sorted(ids)}, _source=["fingerprint"]
        )
        return {d["_id"]: d["_source"]["fingerprint"] for d in docs["docs"] if d.get("found")}
