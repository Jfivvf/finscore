import re
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from langchain_core.messages import HumanMessage
from agents.debate_utils import DEFAULT_ROUNDS, MAX_ROUNDS, MIN_ROUNDS
from agents.synthesizer import THRESHOLD, VERDICT_NEGATIVE, VERDICT_NEUTRAL, VERDICT_POSITIVE
from src.graph import build_graph
import pandas as pd
from src.client import set_credentials
from src.market import add_indicators, last_year, load_daily_candles

# когда подключим реальный RAG по отчётам МСФО надо будет поправить этот текст
DOCS_SOURCE = "Тестовая база агента данных."
MARKET_SOURCE = "Московская биржа, дневные свечи за последний год."
LLM_SOURCE = "GigaChat-2."
AVATAR = {"bull": "🐂", "bear": "🐻"}
TITLE = {"bull": "Бык", "bear": "Медведь"}
SHOW_SOURCES = False

def initial_state(question, rounds):
    return {
        "messages": [HumanMessage(content=question)],
        "task": "",
        "task_type": "irrelevant",
        "ticker": "",
        "quant_data": {},
        "data": [],
        "bull_argument": "",
        "bear_argument": "",
        "final": "",
        "max_rounds": rounds,
        "debate_round": 0,
        "debate_history": [],
    }


def describe(node, payload):
    if node == "orchestrator":
        return f"Оркестратор: тип запроса «{payload.get('task_type')}», задача: {payload.get('task')}"
    if node == "data_agent":
        return f"Агент данных: найдено цитат — {len(payload.get('data', []))}"
    if node == "quant_agent":
        err = (payload.get("quant_data") or {}).get("error")
        return f"Quant: ошибка — {err}" if err else f"Quant: данные биржи по тикеру {payload.get('ticker')} получены"
    if node in ("bull", "bear"):
        entry = payload["debate_history"][-1]
        return f"{TITLE[node]}: раунд {entry['round']}"
    if node == "synthesizer":
        return "Судья: вердикт вынесен"
    return node


def show_turn(entry):
    with st.chat_message("assistant", avatar=AVATAR[entry["side"]]):
        st.markdown(f"**{TITLE[entry['side']]}, раунд {entry['round']}**")
        st.write(entry["text"])


def run_graph(question, rounds):
    result = {"trace": [], "history": [], "final": "", "error": "", "notice": "", "quant": {}, "ticker": "", "citations": []}
    started = time.time()
    with st.status("Агенты работают...", expanded=True) as status:
        try:
            for update in build_graph().stream(initial_state(question, rounds), stream_mode="updates"):
                for node, payload in update.items():
                    payload = payload or {}
                    line = f"{time.time() - started:5.1f} c | {describe(node, payload)}"
                    result["trace"].append(line)
                    st.write(line)
                    if node == "orchestrator" and payload.get("task_type") == "irrelevant":
                        result["notice"] = "Вопрос не относится к финансам и экономике, попробуйте переформулировать."
                    if node == "data_agent":
                        result["citations"] = payload.get("data") or []
                    if node == "quant_agent":
                        result["quant"] = payload.get("quant_data") or {}
                        result["ticker"] = payload.get("ticker", "")
                    if node in ("bull", "bear"):
                        entry = payload["debate_history"][-1]
                        result["history"].append(entry)
                        show_turn(entry)
                    if node == "synthesizer":
                        result["final"] = payload.get("final", "")
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            status.update(label="Ошибка", state="error")
            return result
        status.update(label="Готово", state="complete")
    return result


def score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": "+.2f"},
        title={"text": "Скоринговый балл"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "#333"},
            "steps": [
                {"range": [-1, -THRESHOLD], "color": "#f4a6a6"},
                {"range": [-THRESHOLD, THRESHOLD], "color": "#e6e6e6"},
                {"range": [THRESHOLD, 1], "color": "#a6e3b0"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def load_candles(ticker):
    """Те же дневные свечи и те же формулы, что у quant_agent (src/market.py)."""
    df = load_daily_candles(ticker.upper())
    return last_year(add_indicators(df)) if not df.empty else df


def price_figure(df, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df["begin"], open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name="Цена"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["begin"], y=df["SMA_50"], name="SMA 50", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["begin"], y=df["SMA_200"], name="SMA 200", line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=df["begin"], y=df["volume"], name="Объём", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["begin"], y=df["RSI_14"], name="RSI 14", showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)
    fig.update_yaxes(title_text="Цена, ₽", row=1, col=1)
    fig.update_yaxes(title_text="Объём", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_layout(title=f"{ticker}: котировки", height=650, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=80, b=10))
    #чтобы не было пробелов когда биржа закрыта по выходным
    missing = pd.date_range(df["begin"].min(), df["begin"].max()).difference(df["begin"])
    #missing = pd.date_range(df["begin"].min(), df["begin"].max()).difference(pd.DatetimeIndex(df["begin"]))
    fig.update_xaxes(rangebreaks=[dict(values=missing)])
    return fig


def render_prices(ticker):
    if not ticker or ticker == "UNKNOWN":
        return
    try:
        df = load_candles(ticker)
    except Exception as e:
        st.warning(f"Не удалось загрузить котировки {ticker}: {type(e).__name__}: {e}")
        return
    if df.empty:
        st.warning(f"Нет торговых данных по тикеру {ticker}.")
        return
    st.plotly_chart(price_figure(df, ticker), use_container_width=True)
    st.caption(f"Источник: {MARKET_SOURCE} Свечей: {len(df)}, последняя: {str(df['begin'].iloc[-1])[:16]}.")


def render_sources(result):
    with st.expander("Источники данных", expanded=False):
        st.markdown("**Документы.** " + DOCS_SOURCE)
        if result.get("citations"):
            for i, c in enumerate(result["citations"], 1):
                st.markdown(f"[{i}] {c}")
        if any(not d.get("error") for _, d in quant_items(result.get("quant"))):
            st.markdown(f"**Рынок.** {MARKET_SOURCE}")
        st.markdown("**Модель.** " + LLM_SOURCE)


def quant_items(quant):
    if not quant:
        return []
    if isinstance(quant.get("companies"), dict):
        return list(quant["companies"].items())
    if quant.get("error"):
        return [(None, quant)]
    return [(quant.get("ticker"), quant)]


def render_company(ticker, data):
    if data.get("error"):
        st.warning(f"Данные биржи недоступны ({ticker or 'тикер не определён'}): {data['error']}")
        return
    tech, risk = data.get("technical_indicators", {}), data.get("risk_metrics", {})
    cols = st.columns(6)
    items = [
        ("Цена", data.get("current_price")),
        ("RSI 14", tech.get("RSI_14")),
        ("SMA 50", tech.get("SMA_50")),
        ("SMA 200", tech.get("SMA_200")),
        ("Волатильность 30д, %", risk.get("historical_volatility_30d_pct")),
        ("Просадка от макс. 52н, %", risk.get("max_drawdown_52w_pct")),
    ]
    for col, (label, value) in zip(cols, items):
        col.metric(label, "—" if value is None else value)
    st.caption(f"Метрики на дату: {data.get('date')}")
    render_prices(ticker)


def render_market(quant):
    items = quant_items(quant)
    if not items:
        return
    st.subheader("Данные биржи")
    if len(items) == 1:
        render_company(*items[0])
        return
    tabs = st.tabs([t or "?" for t, _ in items])
    for tab, (ticker, data) in zip(tabs, items):
        with tab:
            render_company(ticker, data)


def render_result(result):
    if result["error"]:
        st.error(result["error"])
    if result.get("notice"):
        st.info(result["notice"])
    render_market(result.get("quant"))
    if result["trace"]:
        with st.expander("Траектория рассуждений агентов", expanded=False):
            st.code("\n".join(result["trace"]), language=None)
    if result["history"]:
        st.subheader("Дебаты")
        for entry in result["history"]:
            show_turn(entry)
    if result["final"]:
        st.subheader("Итоговый отчёт")
        match = re.search(r"Балл:\s*([+-]?\d+(?:\.\d+)?)", result["final"])
        if match:
            st.plotly_chart(score_gauge(float(match.group(1))), use_container_width=True)
            st.caption(
                f"🟥 {VERDICT_NEGATIVE} · "
                f"⬜ {VERDICT_NEUTRAL} · "
                f"🟩 {VERDICT_POSITIVE}"
            )
        st.text(result["final"])
        st.download_button("Скачать отчёт", result["final"], file_name="report.txt", mime="text/plain")
    if SHOW_SOURCES and result["trace"]:
        render_sources(result)


st.set_page_config(page_title="FinScore", page_icon="📈", layout="wide")
st.title("Finscore")

with st.sidebar:
    st.header("Доступ к GigaChat")
    api_key = st.text_input("Authorization key (SberStudio)", type="password",
                            help="Хранится только в памяти вашей сессии, на диск не пишется.")
    scope = st.selectbox("Scope", ["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"])
    if api_key:
        st.session_state["api_key"], st.session_state["scope"] = api_key, scope
    st.caption("Ключ берётся в личном кабинете developers.sber.ru")

question = st.text_input("Введите запрос", placeholder="Например: сравни Сбербанк и ВТБ...")
rounds = st.slider("Раундов дебатов", MIN_ROUNDS, MAX_ROUNDS, DEFAULT_ROUNDS)

if not st.session_state.get("api_key"):
    st.info("Введите ключ GigaChat в боковой панели, чтобы запустить анализ.")

if st.button("Запустить", type="primary", disabled=not (question.strip() and st.session_state.get("api_key"))):
    set_credentials(st.session_state["api_key"], st.session_state.get("scope"))
    st.session_state["result"] = run_graph(question.strip(), rounds)
    st.rerun()
elif "result" in st.session_state:
    render_result(st.session_state["result"])
