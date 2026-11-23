from src.graph import build_graph
from src.state import initial_state
from agents.debate_utils import DEFAULT_ROUNDS

if __name__ == "__main__":
    graph = build_graph()
    task = input()
    state = initial_state(task, DEFAULT_ROUNDS)
    result = graph.invoke(state)
