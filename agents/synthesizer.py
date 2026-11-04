import json
import re
from pathlib import Path
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from client import get_client
from state import State

DIR = Path(__file__).parent

JUDGE_TEMPERATURE = 0.01
JUDGE_MAX_TOKENS = 1200
THRESHOLD = 0.2
EVIDENCE_WEIGHT = {"strong": 1.0, "weak": 0.5, "none": 0.0}

VERDICT_POSITIVE = "Привлекательный"
VERDICT_NEUTRAL = "Нейтральный"
VERDICT_NEGATIVE = "Непривлекательный"
DISCLAIMER = "Это автоматический анализ, а не инвестиционная рекомендация."

MARKUP_RE = re.compile(r"^\s*\d+[.)]\s+|\[\d+\]", re.MULTILINE)


class Argument(BaseModel):
    side: Literal["bull", "bear"]
    text: str
    strength: int = Field(ge=0, le=5)
    evidence: Literal["strong", "weak", "none"]
    source: Optional[int] = None
    note: str = ""


def load_prompt(name):
    """Читает промпт из папки agents."""
    with open(DIR / name, "r", encoding="UTF-8") as f:
        return f.read()


def get_judge_model():
    """GigaChat с настройками судьи."""
    model = get_client(temperature=JUDGE_TEMPERATURE)
    model.max_tokens = JUDGE_MAX_TOKENS
    return model


def extract_json(text):
    """Достаёт JSON из ответа, дописывает потерянную }."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    for candidate in (text[start:end + 1], text[start:].rstrip(" \n`") + "}"):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


def get_citations(state):
    """Цитаты из state в виде строк."""
    data = state.get("data")
    if not isinstance(data, list):
        return []

    citations = []
    for c in data:
        if isinstance(c, dict):
            place = ", ".join(str(x) for x in (c.get("source"), c.get("page")) if x)
            text = str(c.get("text", ""))
            c = f"{text} ({place})" if place else text
        citations.append(str(c))
    return citations


def format_citations(citations):
    """Нумерует цитаты для промпта."""
    message = "Цитаты из документов:\n"
    for i, c in enumerate(citations, 1):
        message += f"{i}: {c}\n"
    return message


def extract_numbers(text):
    """Все числа из текста в одном виде."""
    numbers = set()
    for n in re.findall(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?", MARKUP_RE.sub("", text)):
        n = re.sub(r"[  ]", "", n).replace(",", ".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        numbers.add(n)
    return numbers


def parse_answer(text):
    """Разбирает ответ модели в список аргументов."""
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("arguments"), list):
        return None

    arguments = []
    for raw in data["arguments"]:
        try:
            arguments.append(Argument.model_validate(raw))
        except ValidationError:
            pass
    if not arguments:
        return None
    dropped = len(data["arguments"]) - len(arguments)
    return arguments, str(data.get("missing", "")), dropped


def verify_arguments(arguments, citations):
    """Сверяет ссылки и числа аргументов с цитатами."""
    for a in arguments:
        if a.evidence == "none":
            continue
        if a.source is None:
            if extract_numbers(a.text) - extract_numbers(" ".join(citations)):
                a.note = "числа не совпадают с цитатами"
            else:
                a.evidence = "weak"
                continue
        elif not 1 <= a.source <= len(citations):
            a.note = "нет ссылки на существующую цитату"
        elif extract_numbers(a.text) - extract_numbers(citations[a.source - 1]):
            a.note = "числа не совпадают с цитатой"
        else:
            continue
        a.evidence = "none"


def compute_score(arguments):
    """Балл от -1 до +1 по формуле."""
    total = weights = 0
    for a in arguments:
        w = a.strength * EVIDENCE_WEIGHT[a.evidence]
        total += w if a.side == "bull" else -w
        weights += w
    return round(total / weights, 3) if weights else 0.0


def neutral(summary, attempts=0, failed=False):
    """Нейтральный ответ, когда оценивать нечего."""
    return {
        "verdict": VERDICT_NEUTRAL,
        "score": 0.0,
        "confidence": "низкая",
        "arguments": [],
        "summary": summary,
        "missing": "",
        "attempts": attempts,
        "failed": failed,
        "dropped": 0,
    }


def make_verdict(arguments, missing, citations, attempts, dropped):
    """Вердикт, уверенность и итог по баллу."""
    verify_arguments(arguments, citations)
    score = compute_score(arguments)
    supported = [a for a in arguments if a.strength > 0 and a.evidence != "none"]
    strong = [a for a in supported if a.evidence == "strong"]

    if len(supported) < 2:
        verdict = VERDICT_NEUTRAL
        summary = "Подтверждённых аргументов меньше двух, вывод сделать нельзя."
    elif score > THRESHOLD:
        verdict = VERDICT_POSITIVE
        summary = "Перевес на стороне быка."
    elif score < -THRESHOLD:
        verdict = VERDICT_NEGATIVE
        summary = "Перевес на стороне медведя."
    else:
        verdict = VERDICT_NEUTRAL
        summary = "Стороны почти равны."

    rejected = sum(1 for a in arguments if a.note)
    if rejected:
        summary += f" Не засчитано аргументов: {rejected}."

    if len(supported) < 2 or len(strong) < 2:
        confidence = "низкая"
    elif abs(score) >= 0.6:
        confidence = "высокая"
    else:
        confidence = "средняя"

    return {
        "verdict": verdict,
        "score": score,
        "confidence": confidence,
        "arguments": [a.model_dump() for a in arguments],
        "summary": summary,
        "missing": missing,
        "attempts": attempts,
        "failed": False,
        "dropped": dropped,
    }


def format_final(verdict, citations):
    """Текст ответа для пользователя."""
    lines = [
        f"Вердикт: {verdict['verdict']}",
        f"Балл: {verdict['score']:+.2f} (от -1 до +1), уверенность: {verdict['confidence']}",
        f"Итог: {verdict['summary']}",
    ]
    if verdict["missing"]:
        lines.append(f"Не хватает данных: {verdict['missing']}")
    for side, title in (("bull", "Аргументы за"), ("bear", "Аргументы против")):
        args = [a for a in verdict["arguments"] if a["side"] == side]
        if not args:
            continue
        lines += ["", f"{title}:"]
        for a in args:
            ref = f" [{a['source']}]" if a["source"] else ""
            note = f"; не засчитан: {a['note']}" if a["note"] else ""
            lines.append(f"- {a['text']}{ref} (сила {a['strength']}/5, доказательность: {a['evidence']}{note})")
    if citations:
        lines += ["", "Источники:"]
        lines += [f"[{i}] {c}" for i, c in enumerate(citations, 1)]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def judge(model, bull, bear, citations):
    """Режим choice: оценка аргументов и вердикт."""
    if not citations:
        return neutral("В базе не нашлось цитат, оценить актив не на чем.")
    if not bull and not bear:
        return neutral("Бык и медведь не передали аргументов, сравнивать нечего.")

    messages = [
        SystemMessage(content=load_prompt("synthesizer_choice_prompt.txt")),
        HumanMessage(content=(
            f"{format_citations(citations)}\n"
            f"Аргументы быка:\n{bull or '(нет аргументов)'}\n\n"
            f"Аргументы медведя:\n{bear or '(нет аргументов)'}"
        )),
    ]
    for attempt in (1, 2):
        response = model.invoke(messages)
        parsed = parse_answer(response.content)
        if parsed:
            arguments, missing, dropped = parsed
            return make_verdict(arguments, missing, citations, attempt, dropped)
        messages = messages + [response, HumanMessage(content="Ответ не разобран. Верни только JSON в заданном формате.")]
    return neutral("Не удалось взвесить аргументы: ответ модели не разобран.", attempts=2, failed=True)


def answer_information(model, task, citations):
    """Режим information: ответ по цитатам."""
    if not citations:
        return "В базе не нашлось данных по этому вопросу."

    response = model.invoke([
        SystemMessage(content=load_prompt("synthesizer_information_prompt.txt")),
        HumanMessage(content=f"Вопрос: {task}\n\n{format_citations(citations)}"),
    ])
    result = response.content
    unknown = extract_numbers(result) - extract_numbers(" ".join(citations))
    if unknown:
        result += f"\n\nВнимание: этих чисел нет в цитатах, проверьте их: {', '.join(sorted(unknown))}."
    return result


class RecordingModel:
    def __init__(self, model):
        """Оборачивает модель."""
        self.model = model
        self.responses = []

    def invoke(self, messages):
        """Вызывает модель и запоминает ответ."""
        response = self.model.invoke(messages)
        self.responses.append(response)
        return response


def synthesizer_node(state: State, model=None) -> dict:
    """Узел графа: выбирает режим по task_type."""
    recorder = RecordingModel(model or get_judge_model())
    citations = get_citations(state)

    if state["task_type"] == "information":
        result = answer_information(recorder, state.get("task", ""), citations)
    else:
        verdict = judge(recorder, state.get("bull_argument", ""), state.get("bear_argument", ""), citations)
        result = format_final(verdict, citations)

    print(result, "\n\n")
    return {
        "final": result,
        "messages": recorder.responses
    }
