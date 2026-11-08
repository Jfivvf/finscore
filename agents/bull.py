from agents.debate_utils import run_turn


def bull_node(state, model=None) -> dict:
    round_no = state.get("debate_round", 0) + 1
    update = run_turn(state, "bull", "bull_prompt.txt", round_no, model)
    update["debate_round"] = round_no
    return update
