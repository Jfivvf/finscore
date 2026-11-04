import statistics
from collections import Counter

from agents.synthesizer import format_final, judge
from eval.faithfulness import evaluate_faithfulness


def run_case(model, judge_model, case: dict, runs: int) -> dict:
    """Прогоняет кейс N раз и считает метрики."""
    citations = case["citations"]
    results = [judge(model, case["bull"], case["bear"], citations) for _ in range(runs)]

    verdicts = Counter(r["verdict"] for r in results)
    top_verdict, top_count = verdicts.most_common(1)[0]
    scores = [r["score"] for r in results]
    faithfulness = evaluate_faithfulness(format_final(results[0], citations), citations, judge_model)

    return {
        "name": case["name"],
        "expected": case["expected"],
        "verdict": top_verdict,
        "correct": top_verdict == case["expected"],
        "reproducibility": top_count / runs,
        "score_std": statistics.pstdev(scores),
        "faithfulness": faithfulness["score"],
        "unsupported": faithfulness["unsupported"],
        "json_first_try": sum(r["attempts"] == 1 for r in results) / runs,
        "failed": sum(r["failed"] for r in results) / runs,
    }


def summarize(rows: list[dict]) -> dict:
    """Среднее метрик по всем кейсам."""
    faithfulness = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    return {
        "accuracy": statistics.mean(r["correct"] for r in rows),
        "reproducibility": statistics.mean(r["reproducibility"] for r in rows),
        "score_std": statistics.mean(r["score_std"] for r in rows),
        "faithfulness": statistics.mean(faithfulness) if faithfulness else None,
        "json_first_try": statistics.mean(r["json_first_try"] for r in rows),
        "failed": statistics.mean(r["failed"] for r in rows),
    }


def to_markdown(rows: list[dict], summary: dict, usage: dict, runs: int) -> str:
    """Отчёт report.md."""
    faithfulness = "нет данных" if summary["faithfulness"] is None else f"{summary['faithfulness']:.0%}"
    lines = [
        "# Отчёт по качеству судьи",
        "",
        f"Кейсов: {len(rows)}, прогонов на кейс: {runs}, модель: {usage['model']}.",
        "",
        "| Метрика | Значение | Цель |",
        "|---|---|---|",
        f"| Совпадение с ручной разметкой | {summary['accuracy']:.0%} | ≥ 80% |",
        f"| Воспроизводимость вердикта | {summary['reproducibility']:.0%} | ≥ 80% |",
        f"| Разброс балла (std) | {summary['score_std']:.3f} | ≤ 0.1 |",
        f"| Faithfulness | {faithfulness} | ≥ 90% |",
        f"| Валидный JSON с первой попытки | {summary['json_first_try']:.0%} | ≥ 90% |",
        f"| Не разобрано даже после повтора | {summary['failed']:.0%} | 0% |",
        f"| Токенов на вызов судьи | {usage['tokens_per_call']:.0f} | — |",
        f"| Время вызова судьи, с | {usage['seconds_per_call']:.1f} | — |",
        "",
        "| Кейс | Ожидалось | Вердикт | Воспроизв. | Std | Faithfulness |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        value = "—" if r["faithfulness"] is None else f"{r['faithfulness']:.0%}"
        lines.append(
            f"| {r['name']} | {r['expected']} | {r['verdict']} | {r['reproducibility']:.0%} | {r['score_std']:.3f} | {value} |"
        )
    unsupported = [(r["name"], claim) for r in rows for claim in r["unsupported"]]
    if unsupported:
        lines += ["", "## Неподтверждённые утверждения", ""]
        lines += [f"- {name}: {claim}" for name, claim in unsupported]
    return "\n".join(lines) + "\n"
