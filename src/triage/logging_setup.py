"""Structured JSON logging with a correlation id carried in context variables.

Every log line carries service, run_id, correlation_id, and the active OpenTelemetry trace/span ids, so a run can
be followed across the API, both agents, the retriever and the LLM gateway.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="")
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")
agent_var: contextvars.ContextVar[str] = contextvars.ContextVar("agent", default="")
step_var: contextvars.ContextVar[str] = contextvars.ContextVar("step", default="")

_SERVICE = "triage"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": _SERVICE,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, var in (("run_id", run_id_var), ("correlation_id", correlation_id_var),
                         ("agent", agent_var), ("step", step_var)):
            value = var.get()
            if value:
                payload[key] = value
        try:
            from opentelemetry import trace

            ctx = trace.get_current_span().get_span_context()
            if ctx and ctx.is_valid:
                payload["trace_id"] = format(ctx.trace_id, "032x")
                payload["span_id"] = format(ctx.span_id, "016x")
        except Exception:  # pragma: no cover - tracing optional
            pass
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(service: str, level: str = "INFO", stream=None) -> None:
    global _SERVICE
    _SERVICE = service
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})


def bind(run_id: str = "", correlation_id: str = "", agent: str = "") -> None:
    if run_id:
        run_id_var.set(run_id)
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if agent:
        agent_var.set(agent)
