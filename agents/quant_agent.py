import os
import json
import urllib3
from typing import Dict, Any, TypedDict
from langchain_core.tools import tool
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from moexalgo import Ticker
from langchain_core.tools import tool

# инженерные фиксы
urllib3.disable_warnings()
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

GIGACHAT_B64_KEY = ''

@tool
def fetch_quant_metrics(ticker: str) -> Dict[str, Any]:
    """
    Инструмент для сбора жестких финансовых и рыночных данных с Московской биржи.
    Принимает тикер компании (например, 'SBER', 'LKOH').
    """
    ticker = ticker.upper()
    try:
        stock = Ticker(ticker)
        # Для скользящей средней SMA-200 нам нужен год исторических данных
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        # Получаем данные и конвертируем в DataFrame
        candles = stock.candles(start=start_date, end=end_date)
        if type(candles) is not pd.DataFrame:
            df = pd.DataFrame(candles)
        else:
            df = candles

        if df.empty:
            return {"error": f"Нет торговых данных по тикеру {ticker}."}

        # Оставляем только нужные колонки и сортируем по времени
        df = df[['begin', 'open', 'high', 'low', 'close', 'volume']].copy()
        df.sort_values('begin')

        # 1. Расчет RSI (14 дней)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # 2. Расчет скользящих средних (SMA)
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # 3. Историческая волатильность (за 30 дней)
        # Логарифмическая доходность * корень из 252 торговых дней
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        volatility_30d = df['log_ret'].tail(30).std() * np.sqrt(252) * 100

        # 4. Просадка от 52-недельного максимума
        max_52w = df['high'].max()
        current_price = df['close'].iloc[-1]
        drawdown_pct = ((current_price / max_52w) - 1) * 100

        # Берем последнюю строку для формирования финального ответа
        latest = df.iloc[-1]

        # Защита от NaN (если данных меньше 200 дней)
        def safe_round(val):
            return round(val, 2) if pd.notna(val) else None

        metrics = {
            "ticker": ticker,
            "date": str(latest['begin'])[:10],
            "current_price": safe_round(current_price),
            "technical_indicators": {
                "RSI_14": safe_round(latest['RSI_14']),
                "SMA_50": safe_round(latest['SMA_50']),
                "SMA_200": safe_round(latest['SMA_200']),
                "trend_signal": "Bullish" if latest['SMA_50'] > latest['SMA_200'] else "Bearish"
            },
            "risk_metrics": {
                "historical_volatility_30d_pct": safe_round(volatility_30d),
                "max_drawdown_52w_pct": safe_round(drawdown_pct)
            }
        }
        return metrics

    except Exception as e:
        return {"error": f"Ошибка расчета метрик для {ticker}: {str(e)}"}


llm = GigaChat(
    credentials=GIGACHAT_B64_KEY,
    model="GigaChat-Pro",
    base_url="https://gigachat.devices.sberbank.ru/api/v1",
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    temperature=0.01
)

quant_llm_with_tools = llm.bind_tools([fetch_quant_metrics])


class AgentState(TypedDict):
    user_query: str  # Исходный промпт пользователя
    ticker: str  # Извлеченный тикер
    quant_data: Dict[str, Any]  # Результат работы инструмента


def quant_data_node(state: AgentState) -> dict:
    """Узел графа, принимающий текстовый запрос и вызывающий нужные расчеты."""

    query = state.get("user_query", "Оцени финансовые показатели актива")
    print(f"\n[Quant Agent] Получен запрос из графа: '{query}'")

    sys_prompt = SystemMessage(content=(
        "Ты Quant Agent — строгий финансовый дата-инженер проекта FinScore. "
        "Твоя цель: понять, о какой компании спрашивает пользователь, извлечь ее тикер биржи MOEX "
        "и вызвать инструмент fetch_quant_metrics. Никогда не генерируй текст ответа, используй только инструмент."
    ))
    user_message = HumanMessage(content=query)

    try:
        # LLM анализирует запрос
        ai_response = quant_llm_with_tools.invoke([sys_prompt, user_message])

        # Проверяем, вызвала ли LLM инструмент
        if ai_response.tool_calls:
            tool_call = ai_response.tool_calls[0]
            extracted_ticker = tool_call['args'].get('ticker', 'UNKNOWN')

            print(f"[Quant Agent] ИИ извлек тикер '{extracted_ticker}' и запустил вычисления...")

            # Выполняем функцию с аргументами от LLM
            raw_data = fetch_quant_metrics.invoke(tool_call["args"])
            print("[Quant Agent] Данные успешно получены!")

            return {
                "ticker": extracted_ticker,
                "quant_data": raw_data
            }
        else:
            print("[Quant Agent] ⚠️ Ошибка: ИИ не нашел тикер в запросе.")
            return {"quant_data": {"error": "ИИ не смог преобразовать промпт в вызов функции."}}

    except Exception as e:
        return {"quant_data": {"error": f"Сбой LLM: {str(e)}"}}


# if __name__ == "__main__":
#     test_state = {
#         "user_query": "Привет! Хочу узнать метрики по Сбербанку за последнее время.",
#         "ticker": "",
#         "quant_data": {}
#     }
#
#     print("Запуск тестирования Quant Agent...")
#     final_state = quant_data_node(test_state)
#
#     print("СОСТОЯНИЕ ГРАФА ПОСЛЕ ОТРАБОТКИ АГЕНТА:")
#     print(json.dumps(final_state, indent=4, ensure_ascii=False))
