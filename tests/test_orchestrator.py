from agents.orchestrator import orchestrator_node
from langchain_core.messages import HumanMessage

def basic_check(res):
    if not (isinstance(res, dict) and "task" in res and "task_type" in res):
        return False
    return isinstance(res["task"], str) and isinstance(res["task_type"], str)


def test_1():
    state = {"messages": [HumanMessage(content="Какую музыку ты слушал в школьные годы?")]}
    res = orchestrator_node(state)
    assert basic_check
    task_type = res["task_type"]
    assert task_type == "irrelevant"


def test_2():
    state = {"messages": [HumanMessage(content="Определи, какая компания успешнее: Татнефть или Лукойл")]}
    res = orchestrator_node(state)
    assert basic_check
    task = res["task"]
    task_type = res["task_type"]
    assert task_type == "choice"
    assert "Лукойл" in task and "Татнефть" in task


def test_3():
    state = {"messages": [HumanMessage(content="Сделай скоринг фин активов компаний банковской сферы")]}
    res = orchestrator_node(state)
    assert basic_check
    task = res["task"]
    task_type = res["task_type"]
    assert task_type == "choice"
    assert "банк" in task


def test_4():
    state = {"messages": [HumanMessage(content="Найди последние данные о Газпромбанке")]}
    res = orchestrator_node(state)
    assert basic_check
    task = res["task"]
    task_type = res["task_type"]
    assert task_type == "information"
    assert "Газпромбанк" in task


def test_5():
    state = {"messages": [HumanMessage(content="Сравни ведущие ВУЗы Москвы")]}
    res = orchestrator_node(state)
    assert basic_check
    task_type = res["task_type"]
    assert task_type == "irrelevant"


def test_6():
    state = {"messages": [HumanMessage(content="Какая ситуация в нефтегазовом секторе?")]}
    res = orchestrator_node(state)
    assert basic_check
    task = res["task"]
    task_type = res["task_type"]
    assert task_type == "information"
    assert "нефтегаз" in task