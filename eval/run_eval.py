import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GIGACHAT_MAX_RETRIES", "5")
os.environ.setdefault("GIGACHAT_RETRY_BACKOFF_FACTOR", "2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.synthesizer import get_judge_model
from eval.metrics import run_case, summarize, to_markdown
from eval.tracking import TrackedModel


def main():
    """Замер судьи по cases.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    with open(ROOT / "eval" / "cases.json", "r", encoding="UTF-8") as f:
        cases = json.load(f)

    client = get_judge_model()
    model = TrackedModel(client)
    judge_model = TrackedModel(client)

    rows = [run_case(model, judge_model, case, args.runs) for case in cases]
    usage = {
        "model": client.model,
        "tokens_per_call": model.tokens / max(model.calls, 1),
        "seconds_per_call": model.seconds / max(model.calls, 1),
    }
    report = to_markdown(rows, summarize(rows), usage, args.runs)
    (ROOT / "eval" / "report.md").write_text(report, encoding="UTF-8")
    print(report)


if __name__ == "__main__":
    main()
