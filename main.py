from langchain_core.messages import HumanMessage
from src.graph import build_graph
from agents.debate_utils import DEFAULT_ROUNDS

if __name__ == "__main__":
    graph = build_graph()
    task = input()
    state = {
        "messages": [HumanMessage(content=task)],
        "task": "Sayso",
        "task_type": "irrelevant",
        "ticker": "",
        "quant_data": dict(),
        "data": [],
        "bull_argument": "",
        "bear_argument": "",
        "final": "Sayso",
        "max_rounds": DEFAULT_ROUNDS,
        "debate_round": 0,
        "debate_history": [],
    }
    result = graph.invoke(state)
