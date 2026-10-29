"""The kings row lands first, so the next eval already faces the new king while the upload
runs. After each coronation the older kings' downloads are deleted, each once its mirror is
checked (verify_mirror_before_delete). A gguf king is mirrored twice: its gguf repo at the
coronation, then, when nothing waits to be crowned, an MLX pack of the same weights in a repo
beside it (mlx_once). The pack never delays or fails a coronation: a king whose pack failed
MAX_ATTEMPTS times keeps its download and is left alone until an operator resets
kings.mlx_attempts."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger as log

from common import db, models, reign, states
from common.config import ChainSpec, Quant, Settings, chain_spec, settings
from common.gguf import mlx, read_header
from common.gguf.codes import code_changes
from common.gguf.ptq1 import is_pq2, repack
from common.loop import forever
from promoter import config, speed

Fetch = Callable[[str, Path], Path]
Publish = Callable[[Path, str, str, Quant, dict[str, Path]], str]
Matches = Callable[[Path, str, Quant], bool]
PublishDir = Callable[[Path, str, str, Quant], str]
Snapshot = Callable[..., Path]
UpdateCard = Callable[[str, str], None]


def run_once(
    conn: db.Conn,
    *,
    worker: str,
    s: Settings,
    spec: ChainSpec,
    fetch: Fetch = models.fetch,
    publish: Publish = models.publish,
    matches: Matches = models.mirror_matches,
) -> bool:
    db.heartbeat(conn, worker, queued=states.WON)
    row = db.claim(
        conn, worker=worker, queued=states.WON, running=states.CROWNING, lease_s=config.LEASE_S
    )
    if row is None:
        return False
    try:
        king = db.one(
            conn,
            """
            INSERT INTO kings (submission_id, hotkey, uid, model_uri, family, quant)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (submission_id) DO UPDATE SET hotkey = EXCLUDED.hotkey
            RETURNING id, hf_repo
            """,
            (row["id"], row["hotkey"], row["uid"], row["model_uri"], row["family"], row["quant"]),
        )
        # numbered per pair: the first q2 king is Ingot-I-27B-Q2
        ordinal = db.one(
            conn,
            "SELECT count(*) AS n FROM kings"
            " WHERE submission_id IS NOT NULL AND family = %s AND quant = %s AND id <= %s",
            (row["family"], row["quant"], king["id"]),
        )["n"]
        pair = spec.pair(row["family"], row["quant"])
        title = reign.name(ordinal, pair.king_name)
        if not king["hf_repo"] and s.king_hf_namespace:
            repo_id = f"{s.king_hf_namespace}/{title}"
            model = fetch(row["model_uri"], s.model_cache_dir)
            with tempfile.TemporaryDirectory(dir=s.model_cache_dir) as tmp:
                extra = repacked(model, Path(tmp), title)
                steps = lineage(conn, king["id"], pair)
                text = card(
                    row,
                    title,
                    pair,
                    steps,
                    extra,
                    base_model=spec.teacher_repo,
                    codes=ternary_changes(model, pair),
                )
                publish(model, repo_id, text, pair, extra)
            conn.execute("UPDATE kings SET hf_repo = %s WHERE id = %s", (repo_id, king["id"]))
        elif not king["hf_repo"]:
            log.warning("INGOT_KING_HF_NAMESPACE is empty; {} is not mirrored", title)
    except Exception as exc:
        log.exception("crowning {} failed", row["id"])
        db.fail(
            conn,
            row,
            back_to=states.WON,
            error=f"crowning failed: {exc}",
            max_attempts=config.MAX_ATTEMPTS,
        )
        return True
    db.finish(conn, row["id"], states.CROWNED)
    log.info("crowned {} from submission {} (hotkey {})", title, row["id"], row["hotkey"])
    sweep(conn, s, spec, matches)
    return True


def mlx_once(
    conn: db.Conn,
    *,
    s: Settings,
    spec: ChainSpec,
    fetch: Fetch = models.fetch,
    publish_dir: PublishDir = models.publish_dir,
    snapshot: Snapshot = models.snapshot,
    update_card: UpdateCard = models.update_card,
) -> bool:
    """Mirrors the MLX pack of the oldest gguf king that has none yet: the pack is built from the
    king's gguf in a temporary directory, published beside the gguf repo, recorded, and the gguf
    card gets a link to it. A failure counts an attempt on the king and leaves everything else
    as it was. Returns whether there was a king to do."""
    if not s.king_hf_namespace:
        return False
    rows = conn.execute(
        "SELECT id, submission_id, hotkey, uid, model_uri, hf_repo, family, quant FROM kings"
        " WHERE submission_id IS NOT NULL AND hf_repo IS NOT NULL AND hf_repo_mlx IS NULL"
        " AND mlx_attempts < %s ORDER BY id",
        (config.MAX_ATTEMPTS,),
    ).fetchall()
    king = next((r for r in rows if spec.pair(r["family"], r["quant"]).format == "gguf"), None)
    if king is None:
        return False
    try:
        pair = spec.pair(king["family"], king["quant"])
        ordinal = db.one(
            conn,
            "SELECT count(*) AS n FROM kings"
            " WHERE submission_id IS NOT NULL AND family = %s AND quant = %s AND id <= %s",
            (king["family"], king["quant"], king["id"]),
        )["n"]
        title = reign.name(ordinal, pair.king_name)
        repo_mlx = f"{s.king_hf_namespace}/{models.mlx_repo_name(title)}"
        try:
            model = fetch(king["model_uri"], s.model_cache_dir)
        except Exception:
            # the upload is gone (an expired private store); the mirror holds the same file
            log.exception("fetching {} failed; taking the gguf from {}", title, king["hf_repo"])
            model = snapshot(king["hf_repo"], allow_patterns=f"*-{pair.king_packing}.gguf")
        row = {"hotkey": king["hotkey"], "uid": king["uid"], "id": king["submission_id"]}
        with tempfile.TemporaryDirectory(dir=s.model_cache_dir) as tmp:
            pack = mlx_pack(model, Path(tmp), title, snapshot=snapshot)
            if pack is None:
                raise ValueError(f"{title}: its gguf is not ternary PQ2_0")
            text = card_mlx(
                row,
                title,
                pair,
                lineage(conn, king["id"], pair),
                (pack / mlx.WEIGHTS).stat().st_size,
                gguf_repo=king["hf_repo"],
                base_model=spec.teacher_repo,
                codes=ternary_changes(model, pair),
            )
            publish_dir(pack, repo_mlx, text, pair)
        conn.execute(
            "UPDATE kings SET hf_repo_mlx = %s, mlx_error = NULL WHERE id = %s",
            (repo_mlx, king["id"]),
        )
        update_card(king["hf_repo"], _mlx_link(repo_mlx))
    except Exception as exc:
        log.exception("the MLX pack of king {} failed", king["id"])
        conn.execute(
            "UPDATE kings SET mlx_attempts = mlx_attempts + 1, mlx_error = %s WHERE id = %s",
            (str(exc), king["id"]),
        )
        return True
    log.info("mirrored the MLX pack of {} at {}", title, repo_mlx)
    return True


def repacked(model_dir: Path, tmp: Path, title: str) -> dict[str, Path]:
    """The king's 1.75 bpw PTQ1_0 file in tmp when its gguf is ternary PQ2_0, as every q2 king is:
    Ingot-IV-27B-PTQ1_0.gguf beside Ingot-IV-27B-PQ2_0.gguf."""
    gguf = next(iter(sorted(model_dir.rglob("*.gguf"))), None)
    if gguf is None or not is_pq2(read_header(gguf)):
        return {}
    name = models.gguf_name(title, "PTQ1_0")
    weights = repack(gguf, tmp / name)
    log.info("repacked {} as {} ({:,} weights)", gguf.name, name, weights)
    return {name: tmp / name}


def ternary_changes(model_dir: Path, pair: Quant) -> tuple[int, int] | None:
    """(codes changed, codes compared) of the king's PQ2_0 gguf against genesis's, for a pair
    whose genesis copies someone else's model (pair.upstream), so its card can say how much of
    that model it keeps. None when there is nothing to compare or the comparison failed: the
    card then credits the upstream without the claim."""
    if pair.upstream is None or pair.format != "gguf" or not pair.genesis_file:
        return None
    gguf = next(iter(sorted(model_dir.rglob("*.gguf"))), None)
    try:
        if gguf is None or not is_pq2(read_header(gguf)):
            return None
        genesis = models.hub_file(pair.genesis_repo, pair.genesis_file, pair.genesis_revision)
        changed, total = code_changes(gguf, genesis)
    except Exception:
        log.exception("comparing {} with genesis's codes failed", gguf)
        return None
    log.info("{} changed {:,} of genesis's {:,} ternary codes", gguf.name, changed, total)
    return changed, total


def mlx_pack(
    model_dir: Path, tmp: Path, title: str, *, snapshot: Snapshot = models.snapshot
) -> Path | None:
    """The king's MLX pack in tmp/mlx when its gguf is ternary PQ2_0: its weights converted
    without loss, the rest from Prism's template (common.gguf.mlx), every tensor checked."""
    gguf = next(iter(sorted(model_dir.rglob("*.gguf"))), None)
    if gguf is None or not is_pq2(read_header(gguf)):
        return None
    template = snapshot(mlx.TEMPLATE_REPO, mlx.TEMPLATE_REVISION)
    out = tmp / "mlx"
    weights = mlx.convert(gguf, template, out)
    mlx.verify(out, gguf, template)
    log.info("converted {} to the MLX pack of {} ({:,} weights)", gguf.name, title, weights)
    return out


def sweep(conn: db.Conn, s: Settings, spec: ChainSpec, matches: Matches) -> None:
    """Deletes the downloads of every king but each pair's newest KEEP_KINGS. With
    verify_mirror_before_delete one is deleted only once its mirror matches it; one that does not
    stays, and is checked again after the next coronation. A gguf king stays until its MLX pack
    is mirrored too, however many attempts that took: the pack is built from the gguf."""
    rows = conn.execute(
        "SELECT k.family, k.quant, k.hf_repo, k.hf_repo_mlx, s.model_hash,"
        " row_number() OVER (PARTITION BY k.family, k.quant ORDER BY k.id DESC) AS rank"
        " FROM kings k JOIN submissions s ON s.id = k.submission_id"
    ).fetchall()
    newest = {r["model_hash"] for r in rows if r["rank"] <= config.KEEP_KINGS}
    for r in rows:
        digest = r["model_hash"]
        dest = s.model_cache_dir / digest
        marker = s.model_cache_dir / f"{digest}.verified"
        if digest in newest or not marker.exists():
            continue
        if spec.pair(r["family"], r["quant"]).format == "gguf" and not r["hf_repo_mlx"]:
            log.warning("king {} kept: its MLX pack is not mirrored yet", digest)
            continue
        if s.verify_mirror_before_delete:
            try:
                ok = bool(r["hf_repo"]) and matches(
                    dest, r["hf_repo"], spec.pair(r["family"], r["quant"])
                )
            except Exception:
                log.exception("checking the mirror {} failed", r["hf_repo"])
                ok = False
            if not ok:
                log.warning("king {} kept: its mirror {} does not match", digest, r["hf_repo"])
                continue
        shutil.rmtree(dest, ignore_errors=True)
        marker.unlink(missing_ok=True)
        log.info("deleted older king {} (mirrored at {})", digest, r["hf_repo"])


@dataclass(frozen=True)
class Step:
    """One king's winning duel: its scores and the dethroned king's, on the same samples."""

    kl: float
    kl_king: float
    dist: float
    dist_king: float


def lineage(conn: db.Conn, king_id: int, pair: Quant) -> list[Step]:
    """The winning duels of the pair's kings up to this one, oldest first."""
    rows = conn.execute(
        """
        SELECT e.score_challenger, e.score_king, e.dist_challenger, e.dist_king
        FROM kings k
        JOIN LATERAL (
            SELECT * FROM evals e WHERE e.submission_id = k.submission_id AND e.won
            ORDER BY e.id DESC LIMIT 1
        ) e ON TRUE
        WHERE k.family = %s AND k.quant = %s AND k.id <= %s
        ORDER BY k.id
        """,
        (pair.family, pair.name, king_id),
    ).fetchall()
    return [
        Step(
            *(
                float(r[c])
                for c in ("score_challenger", "score_king", "dist_challenger", "dist_king")
            )
        )
        for r in rows
    ]


def card(
    row: db.Row,
    title: str,
    pair: Quant,
    steps: list[Step],
    extra: dict[str, Path] | None = None,
    *,
    base_model: str | None = None,
    mlx_repo: str | None = None,
    codes: tuple[int, int] | None = None,
) -> str:
    """A king's card. base_model is the teacher's repo, the card's base unless genesis copies
    someone else's model (pair.upstream); codes is (changed, compared) against genesis's
    ternary codes (ternary_changes)."""
    gguf = models.gguf_name(title, pair.king_packing)
    ptq1 = next(iter(extra or {}), None)
    text = _front_matter(pair, base_model) + (
        f"# {title}\n\n"
        "A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a "
        "fixed teacher model and the validator crowns the copy that stays closest to it.\n\n"
        + _upstream(pair, codes, base_model)
        + f"- Submitted by miner hotkey `{row['hotkey']}` (uid {row['uid']})\n"
        f"- Crowned from submission `{row['id']}`\n\n"
    )
    if ptq1:
        text += (
            f"The weights are the miner's upload, byte for byte (`{gguf}`); `{ptq1}` repacks "
            f"them losslessly at 1.75 bits per weight. {_credit(pair)}\n"
            + _files(gguf, ptq1, (extra or {})[ptq1].stat().st_size, pair.weight_limit)
        )
    else:
        text += f"The weights are the miner's upload, byte for byte. {_credit(pair)}\n"
    if mlx_repo:
        text += _mlx_link(mlx_repo)
    if steps:
        text += _scores(steps)
    if pair.format == "gguf":
        text += speed.gguf_section(gguf, ptq1, mlx_repo) + _gguf_card(gguf, pair, ptq1, codes)
    return text


def card_mlx(
    row: db.Row,
    title: str,
    pair: Quant,
    steps: list[Step],
    weights_bytes: int,
    *,
    gguf_repo: str,
    base_model: str | None = None,
    codes: tuple[int, int] | None = None,
) -> str:
    """The MLX pack's card: the gguf king's weights converted without loss, the rest Prism's.
    `title` is the gguf king's; the pack's repo is named from it. With an upstream the base is
    Prism's own MLX pack of it (mlx.TEMPLATE_REPO)."""
    lines = ["---"]
    if pair.upstream:
        lines += [f"base_model: {mlx.TEMPLATE_REPO}", "base_model_relation: finetune"]
    elif base_model:
        lines += [f"base_model: {base_model}", "base_model_relation: quantized"]
    lines += ["library_name: mlx", "pipeline_tag: text-generation", "tags:"]
    lines += [f"- {t}" for t in ("mlx", "apple-silicon", "2-bit", "ternary", "ingot", "bittensor")]
    gguf_hub = f"https://huggingface.co/{gguf_repo}"
    template = mlx.TEMPLATE_REPO
    text = "\n".join([*lines, "---", "", ""]) + (
        f"# {models.mlx_repo_name(title)}\n\n"
        "A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a "
        "fixed teacher model and the validator crowns the copy that stays closest to it. This is "
        f"the MLX pack of [{gguf_repo}]({gguf_hub}): the same ternary weights, written for Apple "
        "Silicon.\n\n"
        + _upstream(pair, codes, base_model)
        + f"- Submitted by miner hotkey `{row['hotkey']}` (uid {row['uid']})\n"
        f"- Crowned from submission `{row['id']}`\n\n"
        "The weights are the miner's upload converted without loss: every 128-weight block keeps "
        "its ternary codes and fp16 scale as MLX affine 2-bit values (scale d, bias −d), and "
        f"every weight decodes to the same number the gguf kernel uses. {_credit(pair)} The "
        "vision tower, config, tokenizer and Hadamard sign vectors are Prism's "
        f"[{template}](https://huggingface.co/{template}), which the competition does not "
        "change.\n"
        "\n## Files\n\n"
        "| file | bits per weight | size |\n| --- | --- | --- |\n"
        f"| `{mlx.WEIGHTS}` | 2.25 (2-bit codes, fp16 scale and bias per 128 weights) "
        f"| {_gb(weights_bytes)} |\n"
    )
    if steps:
        text += _scores(steps)
    text += speed.mlx_section(gguf_repo) + (
        "\n## Running it\n\n"
        "```\n"
        "pip install mlx-lm\n"
        f"mlx_lm.generate --model {gguf_repo.rsplit('/', 1)[0]}/{models.mlx_repo_name(title)}"
        ' --prompt "Hello"\n'
        "```\n\n"
        "The pack loads in any runtime that reads Prism's `prism_hadamard_qwen35` layout, MTPLX "
        f"among them. The gguf twin runs in Prism's llama.cpp fork; see [its card]({gguf_hub}).\n"
    )
    return text


def _mlx_link(mlx_repo: str) -> str:
    """The line the gguf card gets once the MLX pack is mirrored (models.update_card)."""
    return (
        "\nThe same weights as an MLX pack for Apple Silicon: "
        f"[{mlx_repo}](https://huggingface.co/{mlx_repo}).\n"
    )


def _front_matter(pair: Quant, base_model: str | None) -> str:
    """The card's YAML header: with it Hugging Face lists the mirror as a fine-tune of the
    upstream model genesis copies, or else as a quantization of the teacher."""
    gguf = pair.format == "gguf"
    lines = ["---"]
    if pair.upstream:
        lines += [f"base_model: {pair.upstream.repo}", "base_model_relation: finetune"]
    elif base_model:
        lines += [f"base_model: {base_model}", "base_model_relation: quantized"]
    if gguf:
        lines.append("library_name: llama.cpp")
    lines += ["pipeline_tag: text-generation", "tags:"]
    lines += [f"- {tag}" for tag in (["gguf"] if gguf else []) + ["ingot", "bittensor"]]
    return "\n".join([*lines, "---", "", ""])


def _upstream(pair: Quant, codes: tuple[int, int] | None, teacher: str | None) -> str:
    """The paragraph crediting the model genesis copies, with what the miner changed of it: its
    ternary codes kept or the share changed; without a measurement only where the competition
    started."""
    up = pair.upstream
    if up is None:
        return ""
    model = f"[{up.name}](https://huggingface.co/{up.repo})"
    closer = f" to bring it closer to {teacher.rsplit('/', 1)[-1]}" if teacher else ""
    if codes is None:
        return f"Genesis, where the competition started, is {up.author}'s {model}.\n\n"
    changed, total = codes
    if changed == 0:
        return (
            f"It is {up.author}'s {model}, refined: the same ternary codes, with the rest of its "
            f"weights re-tuned by the miner{closer}.\n\n"
        )
    return (
        f"It starts from {up.author}'s {model}: the miner changed {_share(changed, total)} of its "
        f"ternary codes and re-tuned the rest of its weights{closer}.\n\n"
    )


def _share(changed: int, total: int) -> str:
    pct = 100 * changed / total
    return "under 0.01%" if pct < 0.01 else f"{pct:.2f}%"


def _credit(pair: Quant) -> str:
    up = pair.upstream
    if up is None:
        return "Credit belongs to the miner."
    return f"Credit belongs to the miner, and to {up.author} for {up.short}."


def _files(gguf: str, ptq1: str, ptq1_bytes: int, q2_bytes: int | None) -> str:
    """Both files of a q2 king; a q2 king is genesis's size exactly (pair.weight_limit)."""
    return (
        "\n## Files\n\n"
        "| file | bits per weight | size |\n| --- | --- | --- |\n"
        f"| `{ptq1}` | 1.75 | {_gb(ptq1_bytes)} |\n"
        f"| `{gguf}` | 2.125 (the same ternary weights in a 2-bit container) "
        f"| {_gb(q2_bytes)} |\n\n"
        f"Both files hold the same ternary weights. `{ptq1}` is Ingot's lossless repack of the "
        "miner's file into Prism's PTQ1_0 format: every 128-weight block keeps its codes and fp16 "
        "scale, and every weight decodes to the same value. The same repack of Bonsai 2's PQ2_0 "
        "file reproduces Prism's published PTQ1_0 file byte for byte. The 1.75 bpw file is "
        "smaller; the 2-bit file decodes faster on GPUs.\n"
    )


def _scores(steps: list[Step]) -> str:
    """Its winning duel, against the king it beat on the same samples. No gain over genesis: each
    eval scores its own samples, and chaining the ratios along the reign overstated the direct
    gap about threefold when we measured it head to head (2026-10-26)."""
    last = steps[-1]
    return (
        "\n## How close it is to the teacher\n\n"
        "Per answer token, against the BF16 teacher reading the same answer: KL(teacher ‖ model) "
        "of the next-token distributions, and the hidden-state distance |h − h_teacher| / "
        "|h_teacher| averaged over every 8th layer. Lower is closer.\n\n"
        "| | KL | hidden-state distance |\n| --- | --- | --- |\n"
        f"| this model, in its winning eval | {last.kl:.5f} | {last.dist:.4f} |\n"
        f"| the king it beat, same eval | {last.kl_king:.5f} | {last.dist_king:.4f} |\n"
    )


def _gb(n: int | None) -> str:
    return "—" if n is None else f"{n / 1e9:.2f} GB"


def _gguf_card(
    gguf: str, pair: Quant, ptq1: str | None = None, codes: tuple[int, int] | None = None
) -> str:
    """How to run a gguf king: only the fork the eval scored it with runs these files; and the
    genesis vision projector, which the eval never scored, for image input."""
    from eval_service.config import LLAMA_CPP_COMMIT

    fork = "https://github.com/DendriteLabs/llama.cpp"
    hub = f"https://huggingface.co/{pair.genesis_repo}"
    weights = (
        f"its ternary codes {pair.upstream.short}'s and the rest of its weights the miner's"
        if pair.upstream and codes and codes[0] == 0
        else "its weights the miner's"
    )
    text = (
        "\n## Running it\n\n"
        f"`{gguf}` is a gguf with the layout, tokenizer and Hadamard rotation of "
        f"[{pair.genesis_repo}]({hub}) (`{pair.genesis_file}`), {weights}. Stock "
        "llama.cpp does not run it; the Dendrite fork of PrismML's llama.cpp does, at the commit "
        "Ingot scored it with, "
        f"[`{LLAMA_CPP_COMMIT[:7]}`]({fork}/commit/{LLAMA_CPP_COMMIT}):\n\n"
        "```\n"
        f"git clone {fork} && cd llama.cpp && git checkout {LLAMA_CPP_COMMIT}\n"
        "cmake -B build -DGGML_CUDA=ON && cmake --build build -j\n"
        f"build/bin/llama-server -m {gguf} -ngl 99 -fa on --jinja\n"
        "```\n"
    )
    if ptq1:
        text += f"\nEither file runs the same way: pass `-m {ptq1}` for the 1.75 bpw one.\n"
    if drafter := config.DRAFTERS.get(pair.family):
        text += _drafter(gguf, *drafter)
    if pair.vision_files:
        first = pair.vision_files[0]
        vrepo, _, vrev = (pair.vision_repo or "").partition("@")
        vrepo, vrev = vrepo or pair.genesis_repo, vrev or pair.genesis_revision
        vhub = f"https://huggingface.co/{vrepo}"
        files = ", ".join(f"[`{f}`]({vhub}/blob/{vrev}/{f})" for f in pair.vision_files)
        text += (
            "\n### With images\n\n"
            "The file is the text model. For image input, add the vision tower of "
            f"[{vrepo}]({vhub}) as a projector ({files}). It was made for genesis, not "
            "for this model's weights, and Ingot scores the text model only, so image quality "
            "is untested:\n\n"
            "```\n"
            f"hf download {vrepo} {first} --revision {vrev}"
            " --local-dir .\n"
            f"build/bin/llama-server -m {gguf} --mmproj {first} -ngl 99 -fa on --jinja\n"
            "```\n"
        )
    return text


def _drafter(gguf: str, repo: str, file: str) -> str:
    hub = f"https://huggingface.co/{repo}"
    return (
        "\n### Faster generation\n\n"
        f"We trained a DFlash 2 speculative-decoding drafter for this model family, "
        f"[{repo}]({hub}). It proposes several tokens at a time and the model verifies each one, "
        "so it decodes about 1.8x faster in greedy single-stream chat and about 1.3-1.4x when "
        "sampling, at the cost of about 6 GB more VRAM. The same fork runs it:\n\n"
        "```\n"
        f"hf download {repo} {file} --local-dir .\n"
        f"build/bin/llama-server -m {gguf} -md {file} --spec-type draft-dflash "
        "--spec-draft-n-max 5 -ngl 99 -fa on --jinja\n"
        "```\n"
    )


def main() -> None:
    s = settings()
    link = db.Link()
    worker = db.register_worker(link.conn(), "promoter")
    log.info("promoter started — mirroring kings to {}", s.king_hf_namespace or "(nowhere)")
    forever(
        lambda: (
            run_once(link.conn(), worker=worker, s=s, spec=chain_spec())
            or mlx_once(link.conn(), s=s, spec=chain_spec())
        ),
        idle_s=config.IDLE_S,
        on_stop=lambda: db.offline_quietly(link, worker),
    )
