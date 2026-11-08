from typing import TypedDict, Annotated, Dict, Any, List
from langchain_core.messages import BaseMessage
import operator

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    task: str
    task_type : str  # "irrelevant", "information", "choice"
    ticker: str
    quant_data: Dict[str, Any]
    data: List[str]
    bull_argument: str
    bear_argument: str
    final: str
    max_rounds: int
    debate_round: int
    debate_history: Annotated[list[dict], operator.add]
