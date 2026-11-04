from eval.pipeline import citation_fidelity, load_database
import json
from pathlib import Path
from types import SimpleNamespace

from eval.faithfulness import evaluate_faithfulness, split_claims
from eval.metrics import run_case, summarize, to_markdown
from eval.tracking import TrackedModel

CITATIONS = ["Дивиденды 10%", "Долг вырос вдвое"]
FINAL = """Вердикт: Привлекательный
Балл: +0.50 (от -1 до +1), уверенность: средняя

Дивиденды высокие. Долг растёт.

Аргументы за:
- Дивиденды 10% [1] (сила 4/5, доказательность: strong)

Источники:
[1] Дивиденды 10%

Это автоматический анализ, а не инвестиционная рекомендация."""


class Answers:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, messages):
        return SimpleNamespace(content=self.answer, usage_metadata={"total_tokens": 100})


def test_split_claims_skips_service_lines():
    assert split_claims(FINAL) == ["Дивиденды высокие.", "Долг растёт.", "Дивиденды 10%"]


def test_invented_number_is_caught_without_llm():
    result = evaluate_faithfulness("Итог.\n- Дивиденды 50% [1] (сила 5/5, доказательность: strong)", CITATIONS)
    assert result["unsupported"] == ["Дивиденды 50%"]
    assert result["score"] == 0.5


def test_llm_judge_rejects_claims():
    result = evaluate_faithfulness(FINAL, CITATIONS, Answers('{"supported": false, "reason": "нет"}'))
    assert result["score"] == 0


def test_unreadable_judge_answer_is_not_counted():
    result = evaluate_faithfulness(FINAL, CITATIONS, Answers("не знаю"))
    assert result["score"] is None
    assert result["unknown"] == 3


def test_tracked_model_counts_usage():
    model = TrackedModel(Answers("ok"))
    model.invoke([])
    model.invoke([])
    assert (model.calls, model.tokens) == (2, 200)


def test_cases_file_is_valid():
    path = Path(__file__).resolve().parent.parent / "eval" / "cases.json"
    cases = json.loads(path.read_text(encoding="UTF-8"))
    assert len(cases) >= 6
    assert all({"name", "expected", "citations", "bull", "bear"} <= set(c) for c in cases)


def test_report_is_built_from_case_run():
    answer = json.dumps({
        "arguments": [
            {"side": "bull", "text": "Дивиденды 10%", "strength": 5, "evidence": "strong", "source": 1},
            {"side": "bull", "text": "Сильная выплата", "strength": 4, "evidence": "strong", "source": 1},
        ],
        "summary": "Перевес быка.",
    }, ensure_ascii=False)
    case = {"name": "Кейс", "expected": "Привлекательный", "citations": CITATIONS, "bull": "б", "bear": "м"}
    rows = [run_case(Answers(answer), Answers('{"supported": true}'), case, runs=3)]
    summary = summarize(rows)
    assert rows[0]["reproducibility"] == 1
    assert summary["accuracy"] == 1
    report = to_markdown(rows, summary, {"model": "fake", "tokens_per_call": 100, "seconds_per_call": 1}, 3)
    assert "Воспроизводимость вердикта | 100%" in report


def test_citation_fidelity_catches_invented_quote():
    database = "Выручка Лукойла составила 25 млрд рублей. Однако акции упали на 3%;"
    assert citation_fidelity(["Выручка Лукойла составила 25 млрд рублей."], database) == 1
    assert citation_fidelity(["Выручка Лукойла 30 млрд"], database) == 0
    assert citation_fidelity([], database) is None


def test_database_excludes_prompt_example():
    database = load_database()
    assert "ВТБ" in database
    assert "Проанализируй" not in database


def test_rejected_arguments_and_service_lines_are_not_claims():
    final = (
        "Итог: Перевес на стороне медведя.\n"
        "Не хватает данных: нет отчётности.\n"
        "- Дивиденды 25% (сила 4/5, доказательность: none; не засчитан: числа не совпадают с цитатой)\n"
        "- Долг вырос вдвое [2] (сила 4/5, доказательность: strong)"
    )
    assert split_claims(final) == ["Долг вырос вдвое"]


def test_numbering_is_not_paraphrase():
    assert citation_fidelity(['1. "Альфа-банк - 12 % годовых;"'], "Альфа-банк - 12 % годовых, частная компания") == 1
