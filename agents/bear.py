from agents.debate_utils import get_max_rounds, run_turn


def bear_node(state, model=None) -> dict:
    round_no = state.get("debate_round", 1)
    return run_turn(state, "bear", "bear_prompt.txt", round_no, model)


def road_after_bear(state) -> str:
    if state.get("debate_round", 1) < get_max_rounds(state):
        return "bull"
    return "synthesizer"
