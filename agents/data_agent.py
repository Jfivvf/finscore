from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from src.state import State
from src.client import get_client
from pydantic import BaseModel, Field
from typing import List

DIR = Path(__file__).parent
with open(DIR / "data_agent_prompt.txt", "r", encoding="UTF-8") as f:
    prompt = f.read()


class DataAgentQuery(BaseModel):
    """Список цитат по запросу"""
    citations: List[str] = Field(description="Список цитат по запросу")


def data_agent_node(state: State) -> dict:
    agent = create_agent(
        model=get_client(),
        tools=[],
        system_prompt=prompt,
    )
    print("data_agent:")
    response = agent.invoke({"messages": [HumanMessage(content=state["task"])]})["messages"]
    text = response[-1].content
    print(response[-1].usage_metadata, text)

    model = get_client().with_structured_output(
        DataAgentQuery,
        include_raw=True
    )
    response2 = model.invoke([
        SystemMessage(content="Извлеки из текста список цитат"),
        HumanMessage(content=text)
    ])
    data = response2["parsed"]
    print(data, "\n\n", response2["raw"].usage_metadata, "\n\n")
    return {
        "data": data.citations,
        "messages": response + [response2["raw"]]
    }
