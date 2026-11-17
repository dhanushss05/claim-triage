# Everything runs in containers; these targets are shortcuts.
.PHONY: env up down scale ingest reembed eval eval-inprocess demo test test-local loadtest logs ps

env:
	@test -f .env || cp .env.example .env

up: env
	docker compose up -d --build

down:
	docker compose down

scale: env
	docker compose up -d --scale agent-worker=3

ingest:
	docker compose run --rm ingest python -m triage.retrieval.pipeline ingest

reembed:
	docker compose run --rm ingest python -m triage.retrieval.pipeline ingest --reembed

eval:
	docker compose --profile tools run --rm eval --mode api --label stack

eval-inprocess:
	docker compose --profile tools run --rm eval --mode inprocess --label inprocess

demo:
	docker compose --profile tools run --rm --entrypoint python eval scripts/demo.py

test:
	docker compose --profile tools run --rm --entrypoint python eval -m pytest -q tests

# Without Docker: needs `pip install -r requirements-dev.txt` in a virtualenv first.
test-local:
	python -m pytest -q tests

loadtest:
	sh loadtest/run_matrix.sh w$${WORKERS:-1} "5 10 20 40" 90s

logs:
	docker compose logs -f --tail 50

ps:
	docker compose ps
