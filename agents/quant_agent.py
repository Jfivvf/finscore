from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
import urllib3

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from moexalgo import Ticker

from src.client import get_client
from src.state import State

urllib3.disable_warnings()

DIR = Path(__file__).parent
with open(DIR / "quant_prompt.txt", "r", encoding="UTF-8") as f:
    prompt = f.read()


def _safe_round(val: Any) -> float | None:
    """Безопасное округление с конвертацией numpy-типов в стандартный float."""
    return float(round(val, 2)) if pd.notna(val) else None


def _calculate_ticker_metrics(ticker: str) -> Dict[str, Any]:
    """Расчет технических и риск-метрик по одному тикеру через moexalgo."""
    ticker = ticker.strip().upper()
    try:
        stock = Ticker(ticker)
        # Для скользящей средней SMA-200 нужен год исторических данных
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        candles = stock.candles(start=start_date, end=end_date)
        df = candles if isinstance(candles, pd.DataFrame) else pd.DataFrame(candles)

        if df.empty:
            return {"error": f"Нет торговых данных по тикеру {ticker}."}

        # Оставляем нужные колонки и обязательно переприсваиваем отсортированный DataFrame
        df = df[["begin", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("begin")

        # 1. Расчет RSI (14 дней)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # 2. Расчет скользящих средних (SMA)
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_200"] = df["close"].rolling(window=200).mean()

        # 3. Историческая волатильность (за 30 дней)
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        volatility_30d = df["log_ret"].tail(30).std() * np.sqrt(252) * 100

        # 4. Просадка от 52-недельного максимума
        max_52w = df["high"].max()
        current_price = df["close"].iloc[-1]
        drawdown_pct = ((current_price / max_52w) - 1) * 100

        latest = df.iloc[-1]

        sma_50 = latest["SMA_50"]
        sma_200 = latest["SMA_200"]
        if pd.notna(sma_50) and pd.notna(sma_200):
            trend_signal = "Bullish" if sma_50 > sma_200 else "Bearish"
        else:
            trend_signal = None

        return {
            "ticker": ticker,
            "date": str(latest["begin"])[:10],
            "current_price": _safe_round(current_price),
            "technical_indicators": {
                "RSI_14": _safe_round(latest["RSI_14"]),
                "SMA_50": _safe_round(sma_50),
                "SMA_200": _safe_round(sma_200),
                "trend_signal": trend_signal,
            },
            "risk_metrics": {
                "historical_volatility_30d_pct": _safe_round(volatility_30d),
                "max_drawdown_52w_pct": _safe_round(drawdown_pct),
            },
        }

    except Exception as e:
        return {"error": f"Ошибка расчета метрик для {ticker}: {str(e)}"}


@tool
def fetch_quant_metrics(tickers: List[str]) -> Dict[str, Any]:
    """
    Инструмент для сбора технических и рыночных данных с Московской биржи (MOEX).
    Принимает список тикеров компаний (например, ['SBER'] или ['SBER', 'LKOH', 'YDEX']).
    Возвращает словарь с метриками по каждому тикеру.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # Очищаем и убираем дубликаты с сохранением порядка
    unique_tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))

    if not unique_tickers:
        return {"error": "Передан пустой список тикеров."}

    return {ticker: _calculate_ticker_metrics(ticker) for ticker in unique_tickers}


def quant_data_node(state: State) -> dict:
    """Узел графа, извлекающий тикеры из запроса и собирающий по ним тех. метрики."""
    llm = get_client()
    quant_llm_with_tools = llm.bind_tools([fetch_quant_metrics])
    task = state["task"]
    print(f"\n[Quant Agent] Получен запрос из графа: {task}")

    try:
        ai_response = quant_llm_with_tools.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=task)]
        )

        if not ai_response.tool_calls:
            print("[Quant Agent] Ошибка: ИИ не нашел тикеры в запросе.")
            return {
                "tickers": [],
                "quant_data": {"error": "ИИ не смог извлечь тикеры из запроса."},
            }

        extracted_tickers: List[str] = []
        all_metrics: Dict[str, Any] = {}

        # Поддерживаем как один вызов со списком тикеров, так и несколько параллельных вызовов
        for tool_call in ai_response.tool_calls:
            args = tool_call.get("args", {})
            raw_tickers = args.get("tickers") or args.get("ticker")

            if isinstance(raw_tickers, str):
                raw_tickers = [raw_tickers]
            if not raw_tickers:
                continue

            print(f"[Quant Agent] Запуск вычислений для тикеров: {raw_tickers}")
            batch_data = fetch_quant_metrics.invoke({"tickers": raw_tickers})

            if isinstance(batch_data, dict):
                for ticker, metrics in batch_data.items():
                    if ticker != "error" and ticker not in extracted_tickers:
                        extracted_tickers.append(ticker)
                    all_metrics[ticker] = metrics

        print(f"[Quant Agent] Готово. Обработаны тикеры: {extracted_tickers}\n")
        return {
            "tickers": extracted_tickers,
            "quant_data": all_metrics,
        }

    except Exception as e:
        print(f"[Quant Agent] Сбой LLM: {str(e)}")
        return {
            "tickers": [],
            "quant_data": {"error": f"Сбой LLM: {str(e)}"},
        }

