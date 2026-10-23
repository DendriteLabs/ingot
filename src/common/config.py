"""Settings from .env (secrets and deployment values) and chain.toml (what miners and the
validator agree on). Tuning constants live in each service's config.py."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="INGOT_", extra="ignore")

    postgres_user: str = "ingot"
    postgres_password: str = "ingot"
    postgres_db: str = "ingot"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    netuid: int = 97
    network: str = "finney"
    start_block: int = 0
    model_cache_dir: Path = Path.home() / ".cache" / "ingot" / "models"
    dataset_path: Path = Path("eval.jsonl")
    eval_work_dir: Path = Path.home() / ".cache" / "ingot" / "eval"
    artifacts_dir: Path = Path.home() / ".cache" / "ingot" / "artifacts"
    # Hugging Face downloads (genesis, teacher, kings); unset = HF_HOME / ~/.cache/huggingface
    hf_cache_dir: Path | None = None
    hidden_states_dir: Path = Path("/dev/shm/ingot-hidden-states")
    hf_token: str = Field("", validation_alias=AliasChoices("HF_TOKEN", "INGOT_HF_TOKEN"))
    king_hf_namespace: str = "dendriteholdings"
    # the promoter deletes an older king's download only once its mirror matches it byte for byte
    verify_mirror_before_delete: bool = True
    wallet_coldkey: str = ""
    wallet_hotkey: str = ""
    wallet_path: str = ""
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9270
    opensearch_index: str = "ingot_dedup"
    # seeds the fingerprint projections: a new secret needs a freshly seeded bank
    dedup_secret: str = ""
    dedup_device: str = "cuda:0"
    # Comma-separated: several tunnels to it, used in turn (pm2 INGOT_TEACHER_TUNNELS)
    teacher_url: str = ""
    teacher_port: int = 9300
    # one vLLM server per listed GPU; the teacher's are the teacher server's on its own box, or the
    # eval box's when it has no INGOT_TEACHER_URL
    teacher_gpus: str = ""
    king_gpus: str = "0,1,2,3"
    challenger_gpus: str = "4,5,6,7"

    @property
    def database_url(self) -> str:
        auth = f"{quote_plus(self.postgres_user)}:{quote_plus(self.postgres_password)}"
        return f"postgresql://{auth}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    def gpus(self, role: str) -> tuple[int, ...]:
        listed = {"teacher": self.teacher_gpus, "king": self.king_gpus}
        raw = listed.get(role, self.challenger_gpus)
        return tuple(int(g) for g in raw.split(",") if g.strip())

    @property
    def opensearch_url(self) -> str:
        return f"http://{self.opensearch_host}:{self.opensearch_port}"


class R2Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="R2_", extra="ignore")

    account_id: str = ""
    api_token: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    endpoint: str = ""
    private_models_bucket_name: str = ""
    mailbox_bucket_name: str = ""
    artifacts_bucket_name: str = ""
    site_bucket_name: str = ""
    mailbox_signing_key: str = ""
    jurisdiction: str = "default"

    def mailbox_signing_key_bytes(self) -> bytes:
        key = bytes.fromhex(self.mailbox_signing_key)
        if len(key) != 32:
            raise ValueError("R2_MAILBOX_SIGNING_KEY must be 32 bytes of hex")
        return key


FORMATS = ("safetensors", "gguf")
DEDUP_MODES = ("full", "hash")
_HF_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Quant:
    """One competition of a family: a quantization with its format, genesis, limits and king
    name."""

    family: str
    name: str
    format: str  # "safetensors" | "gguf"
    genesis_repo: str
    genesis_revision: str
    # the one file of a gguf genesis
    genesis_file: str | None
    upload_limit: int
    weight_limit: int | None
    king_name: str
    # genesis repo files a crowned king is published with (it was not allowed to ship them)
    publish_files: tuple[str, ...] = ()
    # files that run the vision tower beside a gguf king (llama.cpp --mmproj), in vision_repo
    # (repo@revision) or else the genesis repo
    vision_files: tuple[str, ...] = ()
    vision_repo: str | None = None
    # "full": error fingerprints against the teacher (GPU); "hash": identical weight files only
    dedup: str = "full"


@dataclass(frozen=True)
class ChainSpec:
    # the open pair
    family: str
    quant: str
    teacher_repo: str
    teacher_revision: str
    dataset_repo: str
    dataset_revision: str
    dataset_split: str
    chain_generation: str
    mailbox_url: str
    required_files: tuple[str, ...]
    quants: dict[tuple[str, str], Quant]

    def allows_file(self, name: str, fmt: str) -> bool:
        """A name that may exist in a submission's bucket folder. safetensors: the required files
        or a top-level *.safetensors shard; gguf: a top-level *.gguf (one, see check_files). No
        hidden name (a leading "."): publishing skips those, so a king would go out without them."""
        if "/" in name or name.startswith("."):
            return False
        if fmt == "gguf":
            return name.endswith(".gguf")
        return name in self.required_files or name.endswith(".safetensors")

    def pair(self, family: str, quant: str) -> Quant:
        """The listed pair; an unlisted one is not allowed."""
        if (family, quant) not in self.quants:
            allowed = ", ".join(f"{f}.{q}" for f, q in self.quants)
            raise ValueError(f"{family}.{quant} is not an allowed pair ({allowed})")
        return self.quants[(family, quant)]

    @property
    def open(self) -> Quant:
        """The pair miners may submit now."""
        return self.quants[(self.family, self.quant)]

    def closed(self, family: str, quant: str) -> bool:
        return (family, quant) != (self.family, self.quant)

    def upload_limit(self, family: str, quant: str) -> int:
        """Bytes a miner who declared this pair may upload."""
        return self.pair(family, quant).upload_limit

    def king_name(self, family: str, quant: str) -> str:
        """The suffix a crowned king of this pair carries, e.g. 27B-FP8."""
        return self.pair(family, quant).king_name


@lru_cache
def settings() -> Settings:
    return Settings()


@lru_cache
def r2_settings() -> R2Settings:
    return R2Settings()


@lru_cache
def chain_spec(path: Path = REPO_ROOT / "chain.toml") -> ChainSpec:
    raw = tomllib.loads(path.read_text())
    files = raw["files"]
    return ChainSpec(
        family=raw["competition"]["family"],
        quant=raw["competition"]["active_quant"],
        teacher_repo=raw["teacher"]["repo"],
        teacher_revision=raw["teacher"]["revision"],
        dataset_repo=raw["dataset"]["repo"],
        dataset_revision=raw["dataset"]["revision"],
        dataset_split=raw["dataset"]["split"],
        chain_generation=raw["private_store"]["chain_generation"],
        mailbox_url=raw["private_store"]["mailbox_url"],
        required_files=tuple(files["required"]),
        quants=_quants(raw),
    )


def _quants(raw: dict) -> dict[tuple[str, str], Quant]:
    out: dict[tuple[str, str], Quant] = {}
    for family, quants in raw["quants"].items():
        for quant, c in quants.items():
            where = f"chain.toml [quants.{family}.{quant}]"
            if c.get("format") not in FORMATS:
                raise ValueError(f"{where} format must be one of {', '.join(FORMATS)}")
            for key in ("upload_limit", "weight_limit"):
                value = c.get(key, 1)
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{where} {key} must be a positive byte count")
            if c["format"] == "gguf" and not c.get("genesis_file"):
                raise ValueError(f"{where} needs a genesis_file")
            if not isinstance(c.get("king_name"), str) or not _HF_NAME.fullmatch(c["king_name"]):
                raise ValueError(f"{where} needs a king_name of letters, digits, . _ -")
            if c.get("vision_repo") and "@" not in c["vision_repo"]:
                raise ValueError(f"{where} vision_repo must be repo@revision")
            if c.get("dedup", "full") not in DEDUP_MODES:
                raise ValueError(f"{where} dedup must be one of {', '.join(DEDUP_MODES)}")
            if not c.get("genesis_repo") or not c.get("genesis_revision"):
                raise ValueError(f"{where} needs genesis_repo and genesis_revision")
            out[(family, quant)] = Quant(
                family=family,
                name=quant,
                format=c["format"],
                genesis_repo=c["genesis_repo"],
                genesis_revision=c["genesis_revision"],
                genesis_file=c.get("genesis_file"),
                upload_limit=c["upload_limit"],
                weight_limit=c.get("weight_limit"),
                king_name=c["king_name"],
                vision_files=tuple(c.get("vision_files", [])),
                vision_repo=c.get("vision_repo"),
                dedup=c.get("dedup", "full"),
                publish_files=tuple(
                    c.get(
                        "publish_files",
                        raw["files"]["genesis"] if c["format"] == "safetensors" else [],
                    )
                ),
            )
    names = [q.king_name for q in out.values()]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"chain.toml [quants] king names must be unique: {', '.join(dupes)}")
    competition = raw["competition"]
    if (competition["family"], competition["active_quant"]) not in out:
        open_ = f"{competition['family']}.{competition['active_quant']}"
        raise ValueError(f"chain.toml [competition] opens {open_}, which [quants] does not list")
    return out
