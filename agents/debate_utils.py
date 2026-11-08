import json
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from agents.synthesizer import get_citations, format_citations
from src.client import get_client

DIR = Path(__file__).parent

MIN_ROUNDS = 2
MAX_ROUNDS = 4
DEFAULT_ROUNDS = 3
DEBATE_TEMPERATURE = 0.3
SIDE_TITLE = {"bull": "Бык", "bear": "Медведь"}


def load_prompt(name):
    with open(DIR / name, "r", encoding="UTF-8") as f:
        return f.read()


def get_max_rounds(state):
    try:
        rounds = int(state.get("max_rounds", DEFAULT_ROUNDS))
    except (TypeError, ValueError):
        rounds = DEFAULT_ROUNDS
    return max(MIN_ROUNDS, min(MAX_ROUNDS, rounds))


def join_side(history, side):
    return "\n\n".join(
        f"Раунд {h['round']}: {h['text']}" for h in history if h["side"] == side
    )


def build_messages(system_prompt, state, side, round_no):
    history = state.get("debate_history") or []
    quant = json.dumps(state.get("quant_data") or {}, ensure_ascii=False, indent=2)
    parts = [
        f"Задача пользователя: {state.get('task', '')}",
        "",
        format_citations(get_citations(state)),
        f"Данные от биржи:\n{quant}",
        "",
    ]

    if history:
        parts.append("Ход дебатов:")
        for h in history:
            parts.append(f"[Раунд {h['round']}, {SIDE_TITLE[h['side']]}] {h['text']}")
        parts.append("")

    opponent = [h for h in history if h["side"] != side]
    if opponent:
        parts.append(
            f"Сейчас раунд {round_no}. Ответь на последний довод оппонента "
            f"({SIDE_TITLE[opponent[-1]['side']]}) и добавь новый аргумент."
        )
    else:
        parts.append(f"Сейчас раунд {round_no}. Ты начинаешь дебаты: изложи свою позицию.")

    return [SystemMessage(content=system_prompt), HumanMessage(content="\n".join(parts))]


def run_turn(state, side, prompt_name, round_no, model=None):
    model = model or get_client(temperature=DEBATE_TEMPERATURE)
    response = model.invoke(build_messages(load_prompt(prompt_name), state, side, round_no))
    text = (response.content or "").strip() or "(ответ не получен)"

    entry = {"round": round_no, "side": side, "text": text}
    history = (state.get("debate_history") or []) + [entry]

    print(f"{SIDE_TITLE[side]}, раунд {round_no}:\n{text}\n\n")
    return {
        "debate_history": [entry],
        f"{side}_argument": join_side(history, side),
        "messages": [response],
    }
