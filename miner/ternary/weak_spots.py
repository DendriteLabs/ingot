"""Where a king reads the teacher worst, from the eval records the site publishes: every duel the
king played (as challenger in the eval that crowned it, as king in every eval since), its KL per
domain, source and stratum with each part's share of its score, and the entries it reads worst
across evals. The analysis needs only the network. With --out, the worst samples are rebuilt from
the eval's corpus exactly as the evals cut them, for teacher_data --data.

    python -m miner.ternary.weak_spots [--king Ingot-XXVII-27B-Q2]
    .venv/bin/python -m miner.ternary.weak_spots --out worst.jsonl --n 400"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from collections import defaultdict
from pathlib import Path

SITE = "https://theingot.ai"


def get(site: str, path: str) -> dict:
    req = urllib.request.Request(f"{site}/{path}", headers={"User-Agent": "ingot-miner"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def duels(site: str, king_name: str | None) -> tuple[str, list[tuple[dict, str]]]:
    """The king's name and (eval document, the role it played) for every finished duel."""
    state = get(site, "api/state.json")
    kings = {k["name"]: k for k in state["kings"]}
    king = (
        kings[king_name]
        if king_name
        else next(k for k in kings.values() if k["id"] == state["king"]["id"])
    )
    evals = [(king["eval_id"], "challenger")] if king.get("eval_id") else []
    evals += [
        (e["id"], "king") for e in state["history"] if e["king_id"] == king["id"] and not e["error"]
    ]
    docs = []
    for eval_id, role in sorted(set(evals)):
        doc = get(site, f"api/evals/{eval_id}.json")
        if doc.get("samples"):
            docs.append((doc, role))
    return king["name"], docs


def table(title: str, groups: dict, total: float, min_n: int, limit: int) -> None:
    rows = [(k, v) for k, v in groups.items() if len(v) >= min_n]
    rows.sort(key=lambda kv: -sum(kv[1]))
    print(f"\n{title}")
    print(f"  {'':44} {'samples':>8} {'mean KL':>8} {'share of score':>15}")
    for key, vs in rows[:limit]:
        name = " / ".join(key) if isinstance(key, tuple) else key
        share = 100 * sum(vs) / total
        print(f"  {name[:44]:44} {len(vs):8} {statistics.fmean(vs):8.4f} {share:14.1f}%")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--king", help="a king's name, e.g. Ingot-XXVII-27B-Q2; default: the current king"
    )
    p.add_argument("--site", default=SITE)
    p.add_argument("--out", type=Path, help="write the worst samples, rebuilt from the corpus")
    p.add_argument("--n", type=int, default=400, help="how many entries --out writes")
    a = p.parse_args()

    name, docs = duels(a.site, a.king)
    seen: dict[str, list[tuple[float, str]]] = defaultdict(list)  # entry -> [(KL, eval seed)]
    by_domain, by_source = defaultdict(list), defaultdict(list)
    for doc, role in docs:
        for r in doc["samples"]:
            kl = r["runs"][role]["kl"]
            sid = r["sample_id"]
            seen[sid].append((kl, doc["eval"]["seed"]))
            by_domain[r["domain"]].append(kl)
            by_source[(r["domain"], sid.split("/")[0], sid.rsplit("/", 1)[-1])].append(kl)
    total = sum(sum(v) for v in by_domain.values())
    n = sum(len(v) for v in by_domain.values())
    print(f"{name}: {len(docs)} duels, {n} samples, mean KL {total / n:.4f}")
    table("by domain", by_domain, total, 1, 20)
    table("by domain / source / stratum (at least 10 samples)", by_source, total, 10, 25)

    worst = sorted(seen, key=lambda s: -statistics.fmean(k for k, _ in seen[s]))
    repeats = [s for s in worst if len(seen[s]) > 1][:15]
    if repeats:
        print("\nentries drawn more than once, worst first")
        for s in repeats:
            kls = [k for k, _ in seen[s]]
            print(f"  {s[:70]:70} x{len(kls)}  mean KL {statistics.fmean(kls):.4f}")

    if a.out:
        rebuild(worst[: a.n], seen, a.out)


def rebuild(ids: list[str], seen: dict, out: Path) -> None:
    """The samples as an eval drew them: the corpus row, cut with the seed of its worst duel."""
    from transformers import AutoTokenizer

    from common.config import chain_spec
    from eval_service import dataset, run

    spec = chain_spec()
    tok = AutoTokenizer.from_pretrained(spec.teacher_repo, revision=spec.teacher_revision)
    index = dataset.fetch(spec, Path("eval.jsonl")).index
    if index is None:
        raise SystemExit("the eval's corpus is not a parquet dataset")
    where = {
        sid: (serial, domain, source, language)
        for (domain, language), (sids, serials, sources) in index.rows.items()
        for sid, serial, source in zip(sids, serials, sources, strict=True)
    }
    fits, rows, missing = run.fits(tok), [], 0
    for sid in ids:
        if sid not in where:
            missing += 1  # drawn from an earlier corpus revision
            continue
        serial, domain, source, language = where[sid]
        seed = max(seen[sid])[1]
        sample = dataset._cut(
            dataset._from_index(index, sid, serial, domain, source, language), seed, fits
        )
        if sample is not None:
            rows.append(
                {
                    "id": sid,
                    "domain": domain,
                    "source": source,
                    "language": language,
                    "messages": sample.messages,
                    "tools": sample.tools,
                }
            )
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        f"\n{len(rows)} samples -> {out}"
        + (f" ({missing} not in the current corpus)" if missing else "")
    )


if __name__ == "__main__":
    main()
