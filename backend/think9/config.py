"""Environment-backed settings. One place reads os.environ; everything else takes a Settings."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    database_url: str
    drive_folder_id: str
    google_credentials_json: str | None
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    router_model: str
    embedding_model: str
    reranker_model: str
    coverage_tau: float


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example and fill it in.")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=_required("DATABASE_URL"),
        drive_folder_id=os.environ.get("DRIVE_FOLDER_ID", ""),
        google_credentials_json=os.environ.get("GOOGLE_CREDENTIALS_JSON"),
        llm_api_key=os.environ.get("LLM_API_KEY"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
        router_model=os.environ.get("ROUTER_MODEL", "llama-3.1-8b-instant"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        coverage_tau=float(os.environ.get("COVERAGE_TAU", "0.5")),
    )
