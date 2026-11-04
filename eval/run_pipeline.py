import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GIGACHAT_MAX_RETRIES", "5")
os.environ.setdefault("GIGACHAT_RETRY_BACKOFF_FACTOR", "2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.synthesizer import get_judge_model
from eval.pipeline import load_database, run_question, to_markdown
from src.graph import build_graph


def main():
    """Замер всего графа по pipeline_cases.json."""
    cases = json.loads((ROOT / "eval" / "pipeline_cases.json").read_text(encoding="UTF-8"))
    graph = build_graph()
    client = get_judge_model()
    database = load_database()
    rows = [run_question(graph, case, client, database) for case in cases]
    report = to_markdown(rows, client.model)
    (ROOT / "eval" / "pipeline_report.md").write_text(report, encoding="UTF-8")
    print(report)


if __name__ == "__main__":
    main()
