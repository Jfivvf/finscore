from typing import TypedDict, Annotated, Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage
import operator

def initial_state(question, rounds):
    return {
        "messages": [HumanMessage(content=question)],
        "task": "",
        "task_type": "irrelevant",
        "tickers": [],
        "quant_data": {},
        "data": [],
        "bull_argument": "",
        "bear_argument": "",
        "final": "",
        "max_rounds": rounds,
        "debate_round": 0,
        "debate_history": [],
    }


class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    task: str
    task_type : str  # "irrelevant", "information", "choice"
    tickers: List[str]
    quant_data: Dict[str, Any]
    data: List[str]
    bull_argument: str
    bear_argument: str
    final: str
    max_rounds: int
    debate_round: int
    debate_history: Annotated[list[dict], operator.add]
