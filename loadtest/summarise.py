"""Summarise Locust CSVs: RPS, p50, p95 of the full_run metric against the declared SLO."""
import csv
import glob
import os
import sys

SLO_P95_MS = float(os.environ.get("SLO_P95_MS", "30000"))
d = sys.argv[1] if len(sys.argv) > 1 else "loadtest/results"
rows = []
for path in sorted(glob.glob(os.path.join(d, "*_stats.csv"))):
    label = os.path.basename(path)[:-len("_stats.csv")]
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["Name"] == "full_run":
                rows.append((label, int(r["Request Count"]), int(r["Failure Count"]), float(r["Requests/s"]),
                             float(r["50%"]), float(r["95%"])))
print(f"{'run':<22}{'runs':>6}{'fail':>6}{'runs/s':>9}{'p50 ms':>9}{'p95 ms':>9}  SLO(p95<={SLO_P95_MS:.0f}ms)")
for label, n, f, rps, p50, p95 in rows:
    print(f"{label:<22}{n:>6}{f:>6}{rps:>9.2f}{p50:>9.0f}{p95:>9.0f}  {'met' if p95 <= SLO_P95_MS and f == 0 else 'MISSED'}")
