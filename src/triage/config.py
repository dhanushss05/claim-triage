"""Runtime configuration.

Everything environment-specific (endpoints, credentials, model identifiers, prices) is injected through
environment variables at runtime; see .env.example for the documented list. Nothing here carries a default
for an endpoint, a credential or a model identifier: a missing value fails fast at start-up.
Tunables (timeouts, limits) have safe defaults so that the documented .env.example stays short.
"""
from __future__ import annotations

import os
from functools import lru_cache


class ConfigError(RuntimeError):
    pass


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Required environment variable {name} is not set (see .env.example)")
    return value


def optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def as_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def as_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def as_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Lazily-read settings. Each service only touches the values it needs."""

    # --- service identity -------------------------------------------------------------------------------------
    @property
    def service_name(self) -> str:
        return optional("SERVICE_NAME", "triage")

    @property
    def log_level(self) -> str:
        return optional("LOG_LEVEL", "INFO")

    # --- infrastructure endpoints (injected) ---------------------------------------------------------------------
    @property
    def redis_url(self) -> str:
        return required("REDIS_URL")

    @property
    def retriever_url(self) -> str:
        return required("RETRIEVER_URL")

    @property
    def llm_gateway_url(self) -> str:
        return required("LLM_GATEWAY_URL")

    @property
    def otel_endpoint(self) -> str:
        return optional("OTEL_EXPORTER_OTLP_ENDPOINT")

    # --- LLM provider (gateway only) ---------------------------------------------------------------------------
    @property
    def llm_provider(self) -> str:
        return required("LLM_PROVIDER")

    @property
    def llm_base_url(self) -> str:
        return required("LLM_BASE_URL")

    @property
    def llm_model(self) -> str:
        return required("LLM_MODEL")

    @property
    def llm_api_key(self) -> str:
        return optional("LLM_API_KEY")

    @property
    def llm_price_prompt_per_1m(self) -> float:
        return as_float("LLM_PRICE_PROMPT_PER_1M_USD", 0.0)

    @property
    def llm_price_completion_per_1m(self) -> float:
        return as_float("LLM_PRICE_COMPLETION_PER_1M_USD", 0.0)

    @property
    def llm_daily_budget_usd(self) -> float:
        return as_float("LLM_DAILY_BUDGET_USD", 5.0)

    @property
    def llm_max_rpm(self) -> float:
        """Provider requests per minute the gateway paces itself to (0 = no pacing). Set below the provider quota."""
        return as_float("LLM_MAX_RPM", 0.0)

    @property
    def llm_rate_wait_max_s(self) -> float:
        return as_float("LLM_RATE_WAIT_MAX_S", 15.0)

    @property
    def llm_provider_timeout_s(self) -> float:
        return as_float("LLM_PROVIDER_TIMEOUT_S", 20.0)

    @property
    def breaker_failure_threshold(self) -> int:
        return as_int("BREAKER_FAILURE_THRESHOLD", 5)

    @property
    def breaker_open_seconds(self) -> int:
        return as_int("BREAKER_OPEN_SECONDS", 30)

    # --- LLM client (workers) ------------------------------------------------------------------------------------
    @property
    def llm_call_timeout_s(self) -> float:
        return as_float("LLM_CALL_TIMEOUT_S", 25.0)

    @property
    def llm_max_retries(self) -> int:
        return as_int("LLM_MAX_RETRIES", 2)

    @property
    def retry_base_delay_s(self) -> float:
        return as_float("RETRY_BASE_DELAY_S", 0.5)

    @property
    def tool_timeout_s(self) -> float:
        return as_float("TOOL_TIMEOUT_S", 5.0)

    # --- embeddings / index ----------------------------------------------------------------------------------
    @property
    def embedding_backend(self) -> str:
        return required("EMBEDDING_BACKEND")

    @property
    def embedding_model(self) -> str:
        return optional("EMBEDDING_MODEL")

    @property
    def embedding_allow_fallback(self) -> bool:
        return as_bool("EMBEDDING_ALLOW_FALLBACK", True)

    @property
    def index_dir(self) -> str:
        return optional("INDEX_DIR", "/data/index")

    @property
    def corpus_manifest(self) -> str:
        return optional("CORPUS_MANIFEST", "/app/corpus/manifest.yaml")

    # --- agent limits ------------------------------------------------------------------------------------------
    @property
    def agent_max_steps(self) -> int:
        return as_int("AGENT_MAX_STEPS", 14)

    @property
    def agent_max_tool_calls(self) -> int:
        return as_int("AGENT_MAX_TOOL_CALLS", 12)

    @property
    def agent_max_wall_s(self) -> float:
        return as_float("AGENT_MAX_WALL_S", 90.0)

    @property
    def agent_max_verify_attempts(self) -> int:
        return as_int("AGENT_MAX_VERIFY_ATTEMPTS", 2)

    @property
    def planner_mode(self) -> str:
        return optional("PLANNER_MODE", "hybrid")  # hybrid | heuristic | llm

    @property
    def determination_mode(self) -> str:
        return optional("DETERMINATION_MODE", "hybrid")  # hybrid | rules

    @property
    def enable_modifier_search(self) -> bool:
        return as_bool("ENABLE_MODIFIER_SEARCH", True)

    @property
    def related_relations(self) -> list[str]:
        return [x for x in optional("FIND_RELATED_RELATIONS", "referenced_by_this,cites_this,cites_parent").split(",") if x]

    @property
    def naive_baseline(self) -> bool:
        """Evaluation only: one retrieval pass, model answers, no verification (the section 5 straw man)."""
        return as_bool("NAIVE_BASELINE", False)

    @property
    def derived_k(self) -> int:
        return as_int("DERIVED_K", 3)

    @property
    def worker_concurrency(self) -> int:
        return as_int("WORKER_CONCURRENCY", 4)

    @property
    def run_deadline_s(self) -> float:
        return as_float("RUN_DEADLINE_S", 120.0)

    # --- prompts --------------------------------------------------------------------------------------------
    @property
    def prompts_dir(self) -> str:
        return optional("PROMPTS_DIR", "/app/prompts")

    def prompt_version(self, prompt_id: str) -> str:
        return optional(f"PROMPT_VERSION_{prompt_id.upper()}", "v1")

    # --- API ----------------------------------------------------------------------------------------------
    @property
    def rate_limit_per_minute(self) -> int:
        return as_int("RATE_LIMIT_PER_MINUTE", 30)

    @property
    def rate_limit_burst(self) -> int:
        return as_int("RATE_LIMIT_BURST", 10)

    @property
    def max_inflight_per_client(self) -> int:
        return as_int("MAX_INFLIGHT_PER_CLIENT", 20)

    @property
    def run_cache_enabled(self) -> bool:
        return as_bool("RUN_CACHE_ENABLED", True)

    @property
    def retrieval_cache_enabled(self) -> bool:
        return as_bool("RETRIEVAL_CACHE_ENABLED", True)

    @property
    def run_cache_ttl_s(self) -> int:
        return as_int("RUN_CACHE_TTL_S", 86400)

    @property
    def live_eval_sample_rate(self) -> float:
        return as_float("LIVE_EVAL_SAMPLE_RATE", 0.25)

    @property
    def allow_fault_injection(self) -> bool:
        return as_bool("ALLOW_FAULT_INJECTION", False)

    @property
    def data_dir(self) -> str:
        return optional("DATA_DIR", "/data")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
