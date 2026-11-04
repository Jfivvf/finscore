import json
from types import SimpleNamespace

from agents.synthesizer import (
    Argument,
    VERDICT_NEGATIVE,
    VERDICT_NEUTRAL,
    VERDICT_POSITIVE,
    compute_score,
    extract_numbers,
    get_citations,
    judge,
    synthesizer_node,
    verify_arguments,
)

CITATIONS = ["Дивиденды 10%", "Долг вырос вдвое"]


class FakeModel:

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.answers.pop(0))


def make_state(task_type="choice", **extra):
    state = {
        "task": "Оцени Лукойл",
        "task_type": task_type,
        "data": CITATIONS,
        "bull_argument": "Высокие дивиденды [1]",
        "bear_argument": "Растущий долг [2]",
        "final": "",
    }
    state.update(extra)
    return state


def judgement(*arguments):
    return json.dumps({"arguments": list(arguments), "missing": "Нет данных о прибыли"}, ensure_ascii=False)


def arg(side, strength, evidence, source=1, text=None):
    return {"side": side, "text": text or f"{side} аргумент", "strength": strength, "evidence": evidence, "source": source}


def args(*raw):
    return [Argument.model_validate(a) for a in raw]


def run_final(*answers, **state):
    return synthesizer_node(make_state(**state), model=FakeModel(*answers))["final"]


def test_score_bull_wins():
    assert compute_score(args(arg("bull", 5, "strong"), arg("bear", 1, "strong"))) > 0.2


def test_score_balanced_is_zero():
    assert compute_score(args(arg("bull", 3, "strong"), arg("bear", 3, "strong"))) == 0


def test_unsupported_arguments_do_not_move_score():
    assert compute_score(args(arg("bull", 5, "none"), arg("bear", 2, "strong"))) == -1


def test_numbers_are_normalized():
    assert extract_numbers("1 200,50 и 3.0 и 7") == {"1200.5", "3", "7"}


def test_citations_from_broken_data():
    assert get_citations({"data": "Wrong format"}) == []
    assert get_citations({"data": {"citations": ["a"]}}) == []
    assert get_citations({}) == []


def test_list_numbers_and_refs_are_not_facts():
    assert extract_numbers("2. Выручка 20 млрд [3]") == {"20"}
    assert extract_numbers("1) рост\n2) падение") == set()


def test_numbered_argument_from_thirdtry_keeps_evidence():
    citations = ["Выручка Лукойла составила 25 млрд рублей", "Газпром получил за второй квартал года 20 млрд рублей"]
    arguments = args(arg("bull", 4, "strong", 2, "2. Газпром получил 20 млрд рублей"))
    verify_arguments(arguments, citations)
    assert arguments[0].evidence == "strong"


def test_citations_from_dicts_keep_source():
    data = [{"text": "Долг 5", "source": "МСФО 2025", "page": 12}]
    assert get_citations({"data": data}) == ["Долг 5 (МСФО 2025, 12)"]


def test_positive_verdict():
    final = run_final(judgement(arg("bull", 5, "strong"), arg("bull", 4, "strong", 2), arg("bear", 1, "weak", 2)))
    assert f"Вердикт: {VERDICT_POSITIVE}" in final
    assert "уверенность: высокая" in final
    assert "не инвестиционная рекомендация" in final


def test_negative_verdict():
    final = run_final(judgement(arg("bull", 1, "weak"), arg("bear", 5, "strong", 2), arg("bear", 4, "strong")))
    assert f"Вердикт: {VERDICT_NEGATIVE}" in final


def test_single_supported_argument_is_not_enough():
    final = run_final(judgement(arg("bull", 5, "strong")))
    assert f"Вердикт: {VERDICT_NEUTRAL}" in final
    assert "уверенность: низкая" in final


def test_invented_number_is_not_counted():
    verdict = judge(FakeModel(judgement(
        arg("bull", 5, "strong", 1, "Дивиденды 50%"),
        arg("bear", 3, "strong", 1, "Дивиденды 10%"),
        arg("bear", 3, "strong", 2, "Долг вырос"),
    )), "б", "м", CITATIONS)
    assert verdict["score"] == -1
    assert verdict["verdict"] == VERDICT_NEGATIVE
    assert verdict["arguments"][0]["evidence"] == "none"
    assert verdict["arguments"][0]["note"]


def test_reference_to_missing_citation_is_not_counted():
    verdict = judge(FakeModel(judgement(arg("bull", 5, "strong", 99), arg("bear", 2, "strong"), arg("bear", 2, "weak", 2))), "б", "м", CITATIONS)
    assert verdict["arguments"][0]["evidence"] == "none"
    assert verdict["verdict"] == VERDICT_NEGATIVE


def test_retry_after_bad_json_tells_model_what_is_wrong():
    model = FakeModel("я не JSON", judgement(arg("bull", 5, "strong"), arg("bear", 5, "strong", 2)))
    verdict = judge(model, "б", "м", CITATIONS)
    assert len(model.calls) == 2
    assert len(model.calls[1]) > len(model.calls[0])
    assert verdict["attempts"] == 2


def test_invalid_arguments_are_dropped_and_counted():
    verdict = judge(FakeModel(judgement(
        arg("bull", 5, "strong"), arg("bear", 4, "strong", 2), arg("bull", 99, "strong"), {"side": "???"},
    )), "б", "м", CITATIONS)
    assert len(verdict["arguments"]) == 2
    assert verdict["dropped"] == 2


def test_fallback_when_model_never_returns_json():
    final = run_final("мусор", "снова мусор")
    assert f"Вердикт: {VERDICT_NEUTRAL}" in final
    assert "уверенность: низкая" in final


def test_json_without_last_brace_is_parsed():
    answer = judgement(arg("bull", 4, "strong"), arg("bull", 3, "strong")).rstrip("}")
    assert judge(FakeModel(answer), "б", "м", CITATIONS)["attempts"] == 1


def test_summary_follows_code_not_model():
    fake = arg("bull", 5, "strong", None, "Дивиденды 25%")
    bears = [arg("bear", 4, "strong", 2), arg("bear", 3, "strong", 2)]
    answer = json.dumps({"arguments": [fake, *bears], "missing": "Бык выигрывает"}, ensure_ascii=False)
    result = judge(FakeModel(answer), "б", "м", CITATIONS)
    assert result["verdict"] == VERDICT_NEGATIVE
    assert result["summary"] == "Перевес на стороне медведя. Не засчитано аргументов: 1."


def test_uncited_bear_reasoning_is_weak_not_dropped():
    bulls = [arg("bull", 4, "strong", 1, "Дивиденды 10%"), arg("bull", 4, "strong", 1, "Дивиденды 10%")]
    bears = [arg("bear", 3, "strong", None, "Высокий риск"), arg("bear", 3, "strong", None, "Инфляция съест 10%")]
    result = judge(FakeModel(judgement(*bulls, *bears)), "б", "м", CITATIONS)
    assert result["score"] == 0.455
    assert result["confidence"] == "средняя"
    assert result["arguments"][2]["evidence"] == "weak"


def test_tokens_of_judge_reach_state():
    answer = judgement(arg("bull", 4, "strong"))
    result = synthesizer_node(make_state(), model=FakeModel(answer))
    assert len(result["messages"]) == 1


def test_no_model_call_without_citations():
    model = FakeModel()
    verdict = judge(model, "б", "м", [])
    assert verdict["verdict"] == VERDICT_NEUTRAL
    assert model.calls == []


def test_no_model_call_without_arguments():
    model = FakeModel()
    final = synthesizer_node(make_state(bull_argument="", bear_argument=""), model=model)["final"]
    assert f"Вердикт: {VERDICT_NEUTRAL}" in final
    assert model.calls == []


def test_information_mode():
    model = FakeModel("Дивиденды составляют 10% [1].")
    final = synthesizer_node(make_state(task_type="information"), model=model)["final"]
    assert final == "Дивиденды составляют 10% [1]."
    assert "Оцени Лукойл" in model.calls[0][1].content


def test_information_mode_flags_invented_number():
    final = run_final("Дивиденды составляют 77% [1].", task_type="information")
    assert "77" in final.split("Внимание:")[1]


def test_information_mode_without_citations():
    model = FakeModel()
    final = synthesizer_node(make_state(task_type="information", data="Wrong format"), model=model)["final"]
    assert "не нашлось" in final
    assert model.calls == []
