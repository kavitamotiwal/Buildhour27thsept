from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The refusal gate's fail-closed default.
#
# SIMILARITY_THRESHOLD has an asymmetric failure mode, so the safe direction to fail is not
# obvious. Too low and the bot answers out-of-corpus questions confidently, losing the one
# behaviour this project exists to demonstrate (FR5). Too high and it over-refuses, which
# merely looks unimpressive. A missing threshold used to mean None, which the retriever read
# as "not calibrated yet" and left the gate OPEN: a fresh clone with no .env answered
# everything and still looked like a working app. The default is the value calibrated for the
# bundled corpus, so the gate is live before anyone configures anything.
#
# Re-derive with scripts/calibrate_threshold.py after changing the corpus, EMBEDDING_MODEL,
# CHUNK_SIZE, or CHUNK_OVERLAP. See architecture.md section 5.3.
DEFAULT_SIMILARITY_THRESHOLD = 0.6739

DEFAULTS: dict[str, object] = {
    "EMBEDDING_BACKEND": "auto",
    "EMBEDDING_API_KEY": "",
    "EMBEDDING_BASE_URL": "",
    "EMBEDDING_MODEL": "",
    "LLM_BACKEND": "auto",
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "",
    "LLM_MODEL": "gpt-4o-mini",
    "LLM_TEMPERATURE": 0.2,
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
    "TOP_K": 4,
    "SIMILARITY_THRESHOLD": DEFAULT_SIMILARITY_THRESHOLD,
    "HISTORY_TURNS": 3,
    "RETRIEVAL_HISTORY_TURNS": 10,
    "EMBED_BATCH_SIZE": 32,
    "LLM_TIMEOUT_S": 30,
    "DOCS_DIR": "documents",
    "CHROMA_DIR": "data/chroma",
}

INT_KEYS = (
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "HISTORY_TURNS",
    "RETRIEVAL_HISTORY_TURNS",
    "EMBED_BATCH_SIZE",
    "LLM_TIMEOUT_S",
)
FLOAT_KEYS = ("LLM_TEMPERATURE", "SIMILARITY_THRESHOLD")

HOSTED_DEFAULT_MODEL = "text-embedding-3-small"
LOCAL_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _is_secret(name: str) -> bool:
    return any(marker in name.upper() for marker in _SECRET_MARKERS)


def _read(name: str, default: object) -> object:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    if name in INT_KEYS:
        return int(raw)
    if name in FLOAT_KEYS:
        return float(raw)
    return raw


@dataclass(frozen=True)
class Config:
    EMBEDDING_BACKEND: str = "auto"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_MODEL: str = DEFAULTS["EMBEDDING_MODEL"]
    LLM_BACKEND: str = "auto"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = DEFAULTS["LLM_MODEL"]
    LLM_TEMPERATURE: float = 0.2
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = DEFAULT_SIMILARITY_THRESHOLD
    HISTORY_TURNS: int = 3
    RETRIEVAL_HISTORY_TURNS: int = 10
    EMBED_BATCH_SIZE: int = 32
    LLM_TIMEOUT_S: int = 30
    DOCS_DIR: str = "documents"
    CHROMA_DIR: str = "data/chroma"

    docs_path: Path = field(init=False)
    chroma_path: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "docs_path", self._resolve(self.DOCS_DIR))
        object.__setattr__(self, "chroma_path", self._resolve(self.CHROMA_DIR))

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for f in fields(self):
            if not f.init:
                continue
            out[f.name] = "***" if _is_secret(f.name) and getattr(self, f.name) else getattr(self, f.name)
        return out

    def masked(self) -> str:
        width = max(len(name) for name in self.as_dict())
        return "\n".join(f"{name.ljust(width)} = {value}" for name, value in sorted(self.as_dict().items()))

    @property
    def resolved_embedding_backend(self) -> str:
        if self.EMBEDDING_BACKEND != "auto":
            return self.EMBEDDING_BACKEND
        return "hosted" if self.EMBEDDING_API_KEY else "local"

    @property
    def resolved_llm_backend(self) -> str:
        if self.LLM_BACKEND != "auto":
            return self.LLM_BACKEND
        return "hosted" if self.LLM_API_KEY else "local"


def load_config() -> Config:
    kwargs = {name: _read(name, default) for name, default in DEFAULTS.items()}
    config = Config(**kwargs)
    if not config.EMBEDDING_MODEL:
        default_model = HOSTED_DEFAULT_MODEL if config.resolved_embedding_backend == "hosted" else LOCAL_DEFAULT_MODEL
        object.__setattr__(config, "EMBEDDING_MODEL", default_model)
    return config


CONFIG = load_config()


if __name__ == "__main__":
    print(CONFIG.masked())
