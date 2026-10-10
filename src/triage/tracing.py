"""OpenTelemetry tracing exported over OTLP/HTTP to Jaeger.

Trace context crosses the Redis queue by writing a W3C `traceparent` into each stream message, so one trace spans
API submission, the intake agent, the entitlement agent and every tool/LLM call.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
_configured = False


def setup_tracing(service: str, endpoint: str) -> None:
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces", timeout=2))
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("tracing exporter unavailable: %s", exc)
    trace.set_tracer_provider(provider)
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception:  # pragma: no cover
        pass
    _configured = True


def instrument_fastapi(app) -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ready,metrics")
    except Exception:  # pragma: no cover
        pass


def tracer(name: str = "triage"):
    return trace.get_tracer(name)


def carrier_from_current() -> dict[str, str]:
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


@contextmanager
def attached(carrier: dict[str, str] | None) -> Iterator[None]:
    token = otel_context.attach(extract(carrier or {}))
    try:
        yield
    finally:
        otel_context.detach(token)


def current_trace_id() -> str:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx and ctx.is_valid else ""
