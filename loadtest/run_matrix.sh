#!/usr/bin/env sh
# Step the number of concurrent clients and record Locust CSVs per level.
# Usage (from the repo root): sh loadtest/run_matrix.sh <label> "5 10 20 40" 120s
set -eu
LABEL=${1:-w1}
LEVELS=${2:-"5 10 20 40"}
DURATION=${3:-120s}
for U in $LEVELS; do
  docker compose --profile loadtest run --rm locust locust -f /app/loadtest/locustfile.py --host http://api:8000 \
    --headless -u "$U" -r "$U" -t "$DURATION" --csv "/app/loadtest/results/${LABEL}_u${U}" --only-summary
done
docker compose --profile loadtest run --rm --entrypoint python locust /app/loadtest/summarise.py /app/loadtest/results
