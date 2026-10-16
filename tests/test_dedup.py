"""The dedup rule on a tiny checkpoint with Qwen3.8's layout (one Gated DeltaNet and one gated
attention layer): honest quantizations pass, copies are caught however they are disguised.
Needs the dedup extra (torch)."""

from __future__ import annotations

import shutil

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

from model_dedup.bank import summary_vector  # noqa: E402
from model_dedup.checkpoint import Checkpoint  # noqa: E402
from model_dedup.fingerprint import fingerprint  # noqa: E402
from model_dedup.verdict import decide, similarity  # noqa: E402

P = "model.language_model."
H, V, FF = 64, 300, 128
SECRET = b"x" * 32
GDN_READERS = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
ATTN_READERS = ["q_proj", "k_proj", "v_proj"]


def teacher() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def w(*shape: int) -> torch.Tensor:
        x = torch.randn(*shape, generator=g) * 0.1
        # rows of uneven norm, as in real weights, so matching has something to grip
        return x * torch.rand(shape[0], 1, generator=g).add(0.5) if x.dim() == 2 else x

    t = {"lm_head.weight": w(V, H), P + "embed_tokens.weight": w(V, H), P + "norm.weight": w(H)}
    mixers = {
        0: {
            "linear_attn.in_proj_qkv": (128, H),
            "linear_attn.in_proj_z": (64, H),
            "linear_attn.in_proj_a": (4, H),
            "linear_attn.in_proj_b": (4, H),
            "linear_attn.out_proj": (H, 64),
            "linear_attn.conv1d": (128, 1, 4),
        },
        1: {
            "self_attn.q_proj": (128, H),
            "self_attn.k_proj": (32, H),
            "self_attn.v_proj": (32, H),
            "self_attn.o_proj": (H, 64),
            "self_attn.q_norm": (16,),
            "self_attn.k_norm": (16,),
        },
    }
    for i, mixer in mixers.items():
        layer = f"{P}layers.{i}."
        for name, shape in mixer.items():
            t[layer + name + ".weight"] = w(*shape)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            t[f"{layer}{norm}.weight"] = w(H)
        for proj, shape in (("gate_proj", (FF, H)), ("up_proj", (FF, H)), ("down_proj", (H, FF))):
            t[f"{layer}mlp.{proj}.weight"] = w(*shape)
    return {k: v.bfloat16().float() for k, v in t.items()}  # exactly what gets saved


def quantize(t: dict, seed: int) -> dict:
    """An honest quantization: its own error on the linears; embeddings and lm_head untouched."""
    g = torch.Generator().manual_seed(seed)
    out = {
        k: v + 0.03 * v.std() * torch.randn(v.shape, generator=g) if "proj" in k else v.clone()
        for k, v in t.items()
    }
    return {k: v.bfloat16().float() for k, v in out.items()}


def disguise(t: dict, seed: int) -> dict:
    """Function-preserving: norm gains folded, residual stream rotated and scaled, new norm gains
    in the rotated basis, MLP neurons permuted with up/down scaling, V/O channels scaled."""
    g = torch.Generator().manual_seed(seed)
    t = {k: v.clone() for k, v in t.items()}
    q = torch.linalg.qr(torch.randn(H, H, generator=g))[0]
    c = 1.7
    for i, readers in (
        (0, ["linear_attn." + n for n in GDN_READERS]),
        (1, ["self_attn." + n for n in ATTN_READERS]),
    ):
        layer = f"{P}layers.{i}."
        for norm, rs in (
            (layer + "input_layernorm.weight", readers),
            (layer + "post_attention_layernorm.weight", ["mlp.gate_proj", "mlp.up_proj"]),
        ):
            for r in rs:
                t[layer + r + ".weight"] *= 1 + t[norm]
            t[norm].zero_()
    t["lm_head.weight"] *= 1 + t[P + "norm.weight"]
    t[P + "norm.weight"].zero_()
    for k in t:
        if k.endswith(("o_proj.weight", "out_proj.weight", "down_proj.weight")):
            t[k] = c * q.T @ t[k]
        elif t[k].dim() == 2 and t[k].shape[1] == H:
            t[k] = t[k] @ q
    t[P + "embed_tokens.weight"] *= c
    for i in (0, 1):
        mlp, norm = f"{P}layers.{i}.mlp.", f"{P}layers.{i}.post_attention_layernorm.weight"
        p, s = torch.randperm(FF, generator=g), torch.rand(FF, generator=g) + 0.5
        t[mlp + "gate_proj.weight"] = t[mlp + "gate_proj.weight"][p]
        t[mlp + "up_proj.weight"] = t[mlp + "up_proj.weight"][p] * s[:, None]
        t[mlp + "down_proj.weight"] = t[mlp + "down_proj.weight"][:, p] / s
        gain = torch.rand(H, generator=g) + 0.5
        t[norm] = (1 + t[norm]) * gain - 1
        t[mlp + "gate_proj.weight"] /= gain
        t[mlp + "up_proj.weight"] /= gain
    s = torch.rand(32, generator=g) + 0.5
    t[P + "layers.1.self_attn.v_proj.weight"] *= s[:, None]
    t[P + "layers.1.self_attn.o_proj.weight"] /= s.repeat(2)
    return t


def noised(t: dict, base: dict, k: float, seed: int) -> dict:
    """t plus noise of k times its own error, per matrix."""
    g = torch.Generator().manual_seed(seed)
    return {
        n: v + k * (v - base[n]).norm() / v.numel() ** 0.5 * torch.randn(v.shape, generator=g)
        if "proj" in n
        else v
        for n, v in t.items()
    }


def fp8_blocks(t: dict) -> dict:
    """FP8 e4m3 with one `_scale_inv` per 16 x 16 block, the format validation allows."""
    out = {}
    for k, v in t.items():
        if v.dim() != 2 or "proj" not in k or v.shape[0] % 16 or v.shape[1] % 16:
            out[k] = v
            continue
        scale = v.view(v.shape[0] // 16, 16, v.shape[1] // 16, 16).abs().amax((1, 3)) / 448
        out[k] = (v / scale.repeat_interleave(16, 0).repeat_interleave(16, 1)).to(
            torch.float8_e4m3fn
        )
        out[k + "_scale_inv"] = scale
    return out


def save(path, tensors: dict, dtype=torch.bfloat16):
    path.mkdir(parents=True)
    keep = (torch.float8_e4m3fn,)
    save_file(
        {k: v if v.dtype in keep else v.to(dtype) for k, v in tensors.items()},
        str(path / "model.safetensors"),
    )
    return path


@pytest.fixture(scope="module")
def fps(tmp_path_factory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("models")
    t = teacher()
    a = quantize(t, 1)
    partial = quantize(t, 7)
    partial.update({k: v for k, v in a.items() if "_attn." in k})
    models = {
        "teacher": save(root / "teacher", t),
        "a": save(root / "a", a),
        "b": save(root / "b", quantize(t, 2)),
        "a_disguised": save(root / "a_disguised", disguise(a, 3), torch.float32),
        "a_noise_0.3": save(root / "a_noise_0.3", noised(a, t, 0.3, 4)),
        "a_noise_3": save(root / "a_noise_3", noised(a, t, 3.0, 5)),
        "a_partial": save(root / "a_partial", partial),
        "teacher_disguised": save(root / "teacher_disguised", disguise(t, 6), torch.float32),
        "fp8": save(root / "fp8", fp8_blocks(t)),
    }
    shutil.copytree(models["a"], root / "a_resharded")
    (root / "a_resharded" / "model.safetensors").rename(root / "a_resharded" / "other.safetensors")
    models["a_resharded"] = root / "a_resharded"
    ref = Checkpoint(models["teacher"], "cpu")
    return {n: fingerprint(Checkpoint(p, "cpu"), ref, SECRET) for n, p in models.items()}


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("a_resharded", "COPY"),  # same tensors, new shard names: the file digest differs
        ("a_disguised", "NEAR-COPY"),
        ("a_noise_0.3", "NEAR-COPY"),
        ("a_partial", "NEAR-COPY"),  # only the mixers copied: caught tensor by tensor
        ("b", None),
        ("fp8", None),
    ],
)
def test_copies_are_caught_and_honest_quantizations_pass(fps, candidate, reason):
    verdict = decide(fps[candidate], {"a": fps["a"]})
    assert (verdict.reason, verdict.of) == (reason, "a" if reason else None)


def test_disguises_are_undone(fps):
    s = similarity(fps["a_disguised"], fps["a"])
    assert s["corr"] > 0.999 and s["ratio"] < 0.05
    assert fps["a_disguised"]["residual"]["scale"] == pytest.approx(1.7, rel=1e-4)
    assert fps["teacher_disguised"]["rel_to_teacher"] < 1e-4, "the teacher canonicalizes to itself"


def test_noise_dilutes_a_copy_only_at_the_cost_of_its_error(fps):
    for k, name in ((0.3, "a_noise_0.3"), (3.0, "a_noise_3")):
        assert similarity(fps[name], fps["a"])["corr"] == pytest.approx(
            (1 + k * k) ** -0.5, abs=0.05
        )
    assert fps["a_noise_3"]["rel_to_teacher"] > 3 * fps["a"]["rel_to_teacher"]


def test_summary_vectors_rank_like_the_full_comparison(fps):
    for name in ("a_disguised", "a_noise_0.3", "a_partial", "b"):
        cos = float(summary_vector(fps[name]) @ summary_vector(fps["a"]))
        assert cos == pytest.approx(similarity(fps[name], fps["a"])["corr"], abs=0.1)


def test_fp8_blocks_are_dequantized(fps):
    assert 0.005 < fps["fp8"]["rel_to_teacher"] < 0.05, "an e4m3 rounding error, not a misread"
