import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

FAKE_CLIENT = '''
import json
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

USAGE = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


class FakeChat(BaseChatModel):
    max_tokens: int = 600
    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = AIMessage(content=answer(messages[0].content, messages[-1].content), usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, include_raw=False):
        def call(messages):
            raw = self.invoke(messages)
            return {"raw": raw, "parsed": schema(**json.loads(raw.content)), "parsing_error": None}
        return RunnableLambda(call)


def answer(system, user):
    if system.startswith("Роль: классификатор"):
        kind = "irrelevant" if "погода" in user else "information" if "сколько" in user else "choice"
        return json.dumps({"task": user, "task_type": kind}, ensure_ascii=False)
    if system.startswith("Роль: агент базы данных"):
        return "мало" if "мало" in user else "много"
    if system.startswith("Извлеки из текста"):
        citations = ["Выручка Лукойла составила 25 млрд рублей", "акции Лукойла упали на 3%",
                     "Газпром получил за второй квартал года 20 млрд рублей"]
        return json.dumps({"citations": citations[:1] if user == "мало" else citations}, ensure_ascii=False)
    if system.startswith("Роль: бык"):
        return "БЫК: выручка 25 млрд"
    if system.startswith("Роль: медведь"):
        return "МЕДВЕДЬ: акции упали на 3%"
    if "судья" in system:
        assert "БЫК" in user and "МЕДВЕДЬ" in user, "судья не получил аргументы"
        return json.dumps({"missing": "Нет данных о прибыли.", "arguments": [
            {"side": "bull", "text": "Выручка 25 млрд", "strength": 5, "evidence": "strong", "source": 1},
            {"side": "bull", "text": "Газпром получил 20 млрд", "strength": 5, "evidence": "strong", "source": 3},
            {"side": "bear", "text": "Акции упали на 3%", "strength": 2, "evidence": "strong", "source": 2},
        ]}, ensure_ascii=False)
    if "Утверждение" in user:
        return json.dumps({"supported": True})
    return "Выручка 25 млрд [1]."


def get_client(temperature=0.1):
    return FakeChat()
'''

RUNNER = '''
import contextlib, io, json, sys
from agents.synthesizer import get_judge_model
from eval.pipeline import initial_state, load_database, run_question
from graph import build_graph

question = sys.argv[1]
with contextlib.redirect_stdout(io.StringIO()):
    graph = build_graph()
    result = graph.invoke(initial_state(question))
    row = run_question(graph, {"question": question, "task_type": "choice"}, get_judge_model(), load_database())
print(json.dumps({"final": result["final"], "row": row}, ensure_ascii=False))
'''


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    copy = tmp_path_factory.mktemp("project")
    for name in ("agents", "eval", "graph.py", "state.py", "main.py", "__init__.py"):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy(source, copy / name)
    if "agents\\" in (copy / "agents" / "bull.py").read_text(encoding="UTF-8"):
        subprocess.run(["git", "apply", str(ROOT / "patches" / "skeleton_fixes.patch")], cwd=copy, check=True)
    (copy / "client.py").write_text(FAKE_CLIENT, encoding="UTF-8")
    (copy / "runner.py").write_text(RUNNER, encoding="UTF-8")
    return copy


def run(project, question):
    done = subprocess.run(
        [sys.executable, "runner.py", question], cwd=project, capture_output=True, text=True, encoding="UTF-8"
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_debate_reaches_judge(project):
    final = run(project, "оцени компанию")["final"]
    assert "Вердикт: Привлекательный" in final
    assert "Итог: Перевес на стороне быка." in final
    assert "Не хватает данных: Нет данных о прибыли." in final
    assert "Источники:" in final


def test_information_skips_debate(project):
    assert run(project, "сколько выручка")["final"] == "Выручка 25 млрд [1]."


def test_validator_stops_when_citations_are_few(project):
    assert run(project, "мало данных")["final"] == ""


def test_irrelevant_stops_at_orchestrator(project):
    result = run(project, "какая погода")
    assert result["final"] == ""
    assert result["row"]["route_ok"] is False
    assert result["row"]["fidelity"] is None


def test_pipeline_metrics(project):
    row = run(project, "оцени компанию")["row"]
    assert row["route_ok"] is True
    assert row["fidelity"] == 1
    assert row["faithfulness"] == 1
    assert row["tokens"] == 60
