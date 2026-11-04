import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.synthesizer import DISCLAIMER, extract_json, extract_numbers, format_citations

with open(Path(__file__).parent / "faithfulness_prompt.txt", "r", encoding="UTF-8") as f:
    prompt = f.read()

SKIP = ("Вердикт:", "Балл:", "Итог:", "Не хватает данных:", "Аргументы за:", "Аргументы против:", "Внимание:", DISCLAIMER)


def split_claims(final):
    """Режет ответ на проверяемые утверждения."""
    body = final.split("\nИсточники:")[0]
    claims = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(SKIP) or "не засчитан" in line:
            continue
        line = re.sub(r"\s*\(сила .*\)$", "", line)
        line = re.sub(r"\s*\[\d+\]", "", line).removeprefix("- ")
        claims += [c.strip() for c in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z])", line) if c.strip()]
    return claims


def ask_judge(model, claim, citations):
    """LLM проверяет, подтверждено ли утверждение цитатами."""
    response = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Утверждение: {claim}\n\n{format_citations(citations)}"),
    ])
    result = extract_json(response.content)
    if isinstance(result, dict) and isinstance(result.get("supported"), bool):
        return result["supported"]
    return None


def evaluate_faithfulness(final, citations, model=None):
    """Доля утверждений, подтверждённых цитатами."""
    known = extract_numbers(" ".join(citations))
    supported, unknown = 0, 0
    unsupported = []
    for claim in split_claims(final):
        if extract_numbers(claim) - known:
            unsupported.append(claim)
            continue
        ok = True if model is None else ask_judge(model, claim, citations)
        if ok is None:
            unknown += 1
        elif ok:
            supported += 1
        else:
            unsupported.append(claim)

    checked = supported + len(unsupported)
    return {
        "score": round(supported / checked, 3) if checked else None,
        "checked": checked,
        "unsupported": unsupported,
        "unknown": unknown,
    }
