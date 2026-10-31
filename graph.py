from langgraph.graph import StateGraph, START, END
from state import State
from agents.orchestrator import orchestrator_node, road_after_orchestrator
from agents.data_agent import data_agent_node
from agents.validator import validator_node, road_after_validator
from agents.bear import bear_node
from agents.bull import bull_node
from agents.synthesizer import synthesizer_node


def build_graph():
    workflow = StateGraph(State)
    
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("data_agent", data_agent_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("bear", bear_node)
    workflow.add_node("bull", bull_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges("orchestrator", road_after_orchestrator, {"end_irrelevant": END, "data_agent": "data_agent"})
    workflow.add_edge("data_agent", "validator")
    workflow.add_conditional_edges("validator", road_after_validator, {"end_information_not_found": END, "synthesizer": "synthesizer", "bull": "bull"})
    workflow.add_edge("bull", "bear")
    workflow.add_edge("bear", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()
