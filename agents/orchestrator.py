from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.client import get_client
from src.state import State
from pydantic import BaseModel, Field
from typing import Literal

DIR = Path(__file__).parent
with open(DIR / "orchestrator_prompt.txt", "r", encoding="UTF-8") as f:
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
        HumanMessage(content="Что происходит в ит-сфере?"),
        AIMessage(content='{"task_type": "information", "task": "Что происходит в IT-сфере?IT"}'),
        HumanMessage(content="Кого из компании ты считаешь душой любой встречи?"),
        AIMessage(content='{"task_type": "irrelevant", "task": ""}'),
        HumanMessage(content="Сделай скоринг финансовых активов в нефтяной сфере"),
        AIMessage(content='{"task_type": "choice", "task": "Сделай скоринг финансовых активов в нефтяной сфере"}'),
        HumanMessage(content="Как дела у ит-грандов"),
        AIMessage(content='{"task_type": "information", "task": "Как дела у ит-грандовIT"}'),
        HumanMessage(content=message)
    ])
    
    print("orchestrator:")
    try:
        task = response["parsed"].task
    except:
        task = ""
    try:
        task_type = response["parsed"].task_type
    except:
        task_type = "irrelevant"

    if task_type == "information":
        if "ефтегаз" in task or "нефтян" in task or "газов" in task:
            task += "Посмотри еще Роснефть ROSN, Газпром GAZP, Лукойл LKOH, Татнефть TATN"
        if "банк" in task:
            task += "Посмотри еще ВТБ VTBR, Сбер SBER, Альфа-банк AFKS, Т-банк T, Совкомбанк SVCB"
        if "IT" in task:
            task += "Посмотри еще Яндекс YDEX, Мегафон MFON, Озон OZON, МТС MTSS"
    
    print(task, task_type, response["raw"].usage_metadata, "\n\n")
    return {
        "task": task,
        "task_type": task_type,
        "messages": [response["raw"]]
    }
