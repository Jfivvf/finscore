from langgraph.graph import StateGraph, START, END
from src.state import State
from agents.orchestrator import orchestrator_node
from agents.data_agent import data_agent_node
from agents.quant_agent import quant_data_node
from agents.bear import bear_node, road_after_bear
from agents.bull import bull_node
from agents.synthesizer import synthesizer_node


def road_after_orchestrator(state: State) -> str:
    if state["task_type"] == "irrelevant":
        print("Вопрос не то теме")
        return "end_irrelevant"
    return "data_agent"


def road_after_data_agents(state: State) -> str:
    if state["task_type"] == "choice":
        return "bull"
    return "synthesizer"


def build_graph(checkpointer=None):
    workflow = StateGraph(State)
    
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("data_agent", data_agent_node)
    workflow.add_node("quant_agent", quant_data_node)
    workflow.add_node("bear", bear_node)
    workflow.add_node("bull", bull_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges("orchestrator", road_after_orchestrator, {"end_irrelevant": END, "data_agent": "data_agent"})
    workflow.add_edge("data_agent", "quant_agent")
    workflow.add_conditional_edges("quant_agent", road_after_data_agents, {"synthesizer": "synthesizer", "bull": "bull"})
    workflow.add_edge("bull", "bear")
    # бык, потом медведь, и так max_rounds раз, потом судья
    workflow.add_conditional_edges("bear", road_after_bear, {"bull": "bull", "synthesizer": "synthesizer"})
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=checkpointer)
