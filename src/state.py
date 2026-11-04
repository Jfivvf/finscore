from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    task: str
    task_type : str  # "irrelevant", "information", "choice"
    data: List[str]
    validator_res: bool
    bull_argument: str
    bear_argument: str
    final: str

