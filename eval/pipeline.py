import re
import statistics
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from agents.debate_utils import DEFAULT_ROUNDS
from agents.synthesizer import MARKUP_RE, get_citations, get_sources
from eval.faithfulness import evaluate_faithfulness

DATA_PROMPT = Path(__file__).parent.parent / "agents" / "data_agent_prompt.txt"


def load_database():
    """Тестовая база из промпта агента БД."""
    text = DATA_PROMPT.read_text(encoding="UTF-8")
    return text.split("База данных:", 1)[-1].split("Пример:", 1)[0]


def normalize(text):
    """Текст без разметки, пробелов и регистра."""
    return re.sub(r"\s+", " ", MARKUP_RE.sub("", text)).strip(" .;,\"«»").lower()


def citation_fidelity(citations, database):
    """Доля цитат, дословно найденных в базе."""
    if not citations:
        return None
    base = normalize(database)
    return sum(normalize(c) in base for c in citations) / len(citations)


def count_tokens(messages):
    """Сумма токенов всех агентов."""
    total = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        total += usage.get("total_tokens", 0)
    return total


def initial_state(question):
    """Пустой state графа с вопросом."""
    return {
        "messages": [HumanMessage(content=question)],
        "task": "",
        "task_type": "irrelevant",
        "ticker": "",
        "quant_data": {},
        "data": [],
        "bull_argument": "",
        "bear_argument": "",
        "final": "",
        "max_rounds": DEFAULT_ROUNDS,
        "debate_round": 0,
        "debate_history": [],
    }


def run_question(graph, case, judge_model, database):
    """Прогоняет вопрос через граф и считает метрики."""
    start = time.perf_counter()
    result = graph.invoke(initial_state(case["question"]))
    seconds = time.perf_counter() - start

    citations = get_citations(result)
    faithfulness = None
    if result["final"]:
        faithfulness = evaluate_faithfulness(result["final"], get_sources(result), judge_model)["score"]
    return {
        "question": case["question"],
        "expected": case["task_type"],
        "task_type": result["task_type"],
        "route_ok": result["task_type"] == case["task_type"],
        "fidelity": citation_fidelity(citations, database),
        "faithfulness": faithfulness,
        "tokens": count_tokens(result["messages"]),
        "seconds": seconds,
    }


def mean(values):
    """Среднее без None."""
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def percent(value):
    """Число в проценты или «нет данных»."""
    return "нет данных" if value is None else f"{value:.0%}"


def to_markdown(rows, model_name):
    """Отчёт pipeline_report.md."""
    lines = [
        "# Отчёт по всему пайплайну",
        "",
        f"Вопросов: {len(rows)}, модель: {model_name}.",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Точность маршрута (тип запроса) | {percent(mean([r['route_ok'] for r in rows]))} |",
        f"| Цитаты дословно из базы | {percent(mean([r['fidelity'] for r in rows]))} |",
        f"| Faithfulness итогового ответа | {percent(mean([r['faithfulness'] for r in rows]))} |",
        f"| Токенов на запрос | {statistics.mean(r['tokens'] for r in rows):.0f} |",
        f"| Время на запрос, с | {statistics.mean(r['seconds'] for r in rows):.1f} |",
        "",
        "| Вопрос | Ожидался тип | Получен | Цитаты | Faithfulness | Токены | Время, с |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['question']} | {r['expected']} | {r['task_type']} | {percent(r['fidelity'])} "
            f"| {percent(r['faithfulness'])} | {r['tokens']} | {r['seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"
