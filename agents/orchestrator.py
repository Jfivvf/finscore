from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from client import get_client
from state import State
from pydantic import BaseModel, Field
from typing import Literal

with open(r"agents\orchestrator_prompt.txt", "r", encoding="UTF-8") as f:
    prompt = f.read()


class OrchestratorQuery(BaseModel):
    """Определение задачи и типа запроса"""
    task: str = Field(description="Описание задачи")
    task_type: Literal["irrelevant", "information", "choice"] = Field(description="Тип запроса")


def orchestrator_node(state: State) -> dict:
    """Определение задачи и типа запроса"""
    model = get_client().with_structured_output(
        OrchestratorQuery,
        include_raw=True
    )
    message = state["messages"][-1].content
    response = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=message)
    ])
    
    print("orchestrator:")
    task = response["parsed"].task
    task_type = response["parsed"].task_type
    print(task, task_type, response["raw"].usage_metadata, "\n\n")
    return {
        "task": task,
        "task_type": task_type,
        "messages": [response["raw"]]
    }


def road_after_orchestrator(state: State) -> str:
    print("road_after_orchestrator", state["task_type"])
    if state["task_type"] == "irrelevant":
        print("Вопрос не то теме")
        return "end_irrelevant"
    return "data_agent"
