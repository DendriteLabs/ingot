"""Settings from .env (secrets and deployment values) and chain.toml (what miners and the
validator agree on). Tuning constants live in each service's config.py."""

from __future__ import annotations

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
    wallet_coldkey: str = ""
    wallet_hotkey: str = ""
    wallet_path: str = ""
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9270
    opensearch_index: str = "ingot_dedup"
    # seeds the fingerprint projections: a new secret needs a freshly seeded bank
    dedup_secret: str = ""
    dedup_device: str = "cuda:0"
    # eval box: the teacher served by another box (ingot-teacher-server); empty = its own GPUs
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


@dataclass(frozen=True)
class ChainSpec:
    genesis_repo: str
    genesis_revision: str
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
    genesis_files: tuple[str, ...]
    upload_limits: dict[tuple[str, str], int]
    king_names: dict[tuple[str, str], str]

    def allows_file(self, name: str) -> bool:
        """A name that may exist in a submission's bucket folder: the required files or a
        top-level *.safetensors shard."""
        if "/" in name:
            return False
        return name in self.required_files or name.endswith(".safetensors")

    def upload_limit(self, family: str, quant: str) -> int:
        """Bytes a miner who declared this pair may upload; an unlisted pair is not allowed."""
        if (family, quant) not in self.upload_limits:
            allowed = ", ".join(f"{f}.{q}" for f, q in self.upload_limits)
            raise ValueError(f"{family}.{quant} is not an allowed pair ({allowed})")
        return self.upload_limits[(family, quant)]

    def king_name(self, family: str, quant: str) -> str:
        """The suffix a crowned king of this pair carries, e.g. 27B-FP8."""
        return self.king_names[(family, quant)]


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
        genesis_repo=raw["genesis"]["repo"],
        genesis_revision=raw["genesis"]["revision"],
        family=raw["genesis"]["family"],
        quant=raw["genesis"]["quant"],
        teacher_repo=raw["teacher"]["repo"],
        teacher_revision=raw["teacher"]["revision"],
        dataset_repo=raw["dataset"]["repo"],
        dataset_revision=raw["dataset"]["revision"],
        dataset_split=raw["dataset"]["split"],
        chain_generation=raw["private_store"]["chain_generation"],
        mailbox_url=raw["private_store"]["mailbox_url"],
        required_files=tuple(files["required"]),
        genesis_files=tuple(files["genesis"]),
        upload_limits=_upload_limits(raw),
        king_names=_king_names(raw),
    )


def _upload_limits(raw: dict) -> dict[tuple[str, str], int]:
    limits = {
        (family, quant): limit
        for family, quants in raw["upload_limits"].items()
        for quant, limit in quants.items()
    }
    if bad := [f"{f}.{q}" for (f, q), v in limits.items() if not isinstance(v, int) or v < 1]:
        raise ValueError(f"chain.toml [upload_limits] needs positive byte counts: {bad}")
    genesis = (raw["genesis"]["family"], raw["genesis"]["quant"])
    if genesis not in limits:
        raise ValueError(f"chain.toml [upload_limits] has no entry for genesis {'.'.join(genesis)}")
    return limits


def _king_names(raw: dict) -> dict[tuple[str, str], str]:
    names = {
        (family, quant): name
        for family, quants in raw["king_names"].items()
        for quant, name in quants.items()
    }
    if bad := [f"{f}.{q}" for (f, q), v in names.items() if not isinstance(v, str) or not v]:
        raise ValueError(f"chain.toml [king_names] needs non-empty names: {bad}")
    if names.keys() != _upload_limits(raw).keys():
        raise ValueError("chain.toml [king_names] must list the same pairs as [upload_limits]")
    return names
