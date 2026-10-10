# One Dockerfile, several targets, so each service gets only the dependencies its resource profile needs.
#   app        api, intake-agent, agent-worker, llm-gateway, llm-stub   (no ML libraries)
#   retriever  retriever, ingest                                        (adds CPU torch + sentence-transformers)
#   tools      eval, locust                                              (adds locust, pytest)
FROM python:3.11-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# Optional: corporate/proxy root certificates dropped into docker/certs/*.crt are trusted at build and run time.
COPY docker/certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates >/dev/null 2>&1 || true
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM deps AS deps-embeddings
# INSTALL_EMBEDDINGS=false skips torch/sentence-transformers (~1 GB) for low-resource machines or offline builds;
# then set EMBEDDING_BACKEND=hashing in .env.
ARG INSTALL_EMBEDDINGS=true
COPY requirements-embeddings.txt .
RUN if [ "$INSTALL_EMBEDDINGS" = "true" ]; then pip install -r requirements-embeddings.txt; fi

FROM deps AS app
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY corpus/ /app/corpus/
COPY contracts/ /app/contracts/
ENV PYTHONPATH=/app/src:/app PROMPTS_DIR=/app/prompts CORPUS_MANIFEST=/app/corpus/manifest.yaml CONTRACTS_DIR=/app/contracts
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /data
USER app

FROM deps-embeddings AS retriever
COPY src/ /app/src/
COPY prompts/ /app/prompts/
COPY corpus/ /app/corpus/
COPY contracts/ /app/contracts/
ENV PYTHONPATH=/app/src:/app PROMPTS_DIR=/app/prompts CORPUS_MANIFEST=/app/corpus/manifest.yaml HF_HOME=/data/hf
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown -R app /data
USER app

FROM app AS tools
USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY eval/ /app/eval/
COPY loadtest/ /app/loadtest/
COPY tests/ /app/tests/
COPY scripts/ /app/scripts/
RUN chown -R app /app/eval /app/loadtest
USER app
