from langchain_core.messages import HumanMessage
from graph import build_graph

if __name__ == "__main__":
    graph = build_graph()
    task = input()
    state = {
        "messages": [HumanMessage(content=task)],
        "task": "Sayso",
        "task_type": "irrelevant",
        "validator_res": False,
        "data": [],
        "bull_argument": "",
        "bear_argument": "",
        "final": "Sayso"
    }
    result = graph.invoke(state)
