"""Shared FastAPI plumbing: correlation id, request metrics, /metrics, structured logging, tracing."""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..config import get_settings
from ..logging_setup import correlation_id_var, run_id_var, setup_logging
from ..metrics import HTTP_LATENCY, HTTP_REQUESTS
from ..tracing import instrument_fastapi, setup_tracing


def make_app(service: str, **kw) -> FastAPI:
    s = get_settings()
    setup_logging(service, s.log_level)
    setup_tracing(service, s.otel_endpoint)
    app = FastAPI(title=service, **kw)
    instrument_fastapi(app)
    app.state.started = time.time()

    @app.middleware("http")
    async def correlation_and_metrics(request: Request, call_next):
        cid = request.headers.get("X-Correlation-Id") or request.headers.get("X-Run-Id") or uuid.uuid4().hex[:16]
        correlation_id_var.set(cid)
        if request.headers.get("X-Run-Id"):
            run_id_var.set(request.headers["X-Run-Id"])
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Correlation-Id"] = cid
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            if path not in ("/metrics", "/health"):
                HTTP_REQUESTS.labels(service, path, request.method, str(status)).inc()
                HTTP_LATENCY.labels(service, path).observe(time.perf_counter() - t0)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
