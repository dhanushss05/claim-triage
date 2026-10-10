"""Writes the published JSON Schemas into contracts/ (run: make contracts)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from triage.contracts import ClaimRequest, Feedback, IntakeHandoff, RunAccepted, RunResult  # noqa: E402

for name, model in {"claim_request": ClaimRequest, "run_accepted": RunAccepted, "run_result": RunResult,
                    "feedback": Feedback, "intake_handoff": IntakeHandoff}.items():
    with open(os.path.join(ROOT, "contracts", f"{name}.schema.json"), "w") as fh:
        json.dump(model.model_json_schema(), fh, indent=2)
    print("wrote", name)
