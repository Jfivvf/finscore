from langchain_core.messages import AIMessage
from agents.bull import bull_node
from agents.bear import bear_node, road_after_bear
from agents.debate_utils import get_max_rounds


class FakeModel:
    def __init__(self):
        self.calls = []
    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=f"ответ {len(self.calls)}")


def apply(state, update):
    for key, value in update.items():
        if key in ("debate_history", "messages"):
            state[key] = state.get(key, []) + value
        else:
            state[key] = value


def run_debate(max_rounds):
    state = {"task": "Оцени SBER", "data": ["Выручка 100 млрд (отчёт, стр. 1)"],
             "max_rounds": max_rounds, "debate_round": 0, "debate_history": [], "messages": []}
    model = FakeModel()
    while True:
        apply(state, bull_node(state, model))
        apply(state, bear_node(state, model))
        if road_after_bear(state) == "synthesizer":
            return state, model


def test_bull_goes_first_and_sides_alternate():
    state, _ = run_debate(3)
    assert [(h["round"], h["side"]) for h in state["debate_history"]] == [
        (1, "bull"), (1, "bear"), (2, "bull"), (2, "bear"), (3, "bull"), (3, "bear")]


def test_rounds_are_clamped_to_2_4():
    assert get_max_rounds({"max_rounds": 1}) == 2
    assert get_max_rounds({"max_rounds": 9}) == 4
    assert get_max_rounds({}) == 3
    assert len(run_debate(1)[0]["debate_history"]) == 4


def test_arguments_for_synthesizer_are_filled_separately():
    state, _ = run_debate(2)
    assert "Раунд 1" in state["bull_argument"] and "Раунд 2" in state["bull_argument"]
    assert state["bull_argument"] != state["bear_argument"]


def test_bear_sees_bull_argument():
    _, model = run_debate(2)
    assert "ответ 1" in model.calls[1][1].content 
