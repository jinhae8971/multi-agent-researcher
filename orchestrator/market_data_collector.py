"""
market_data_collector.py — 도메인별 정량 시장 데이터 수집 모듈

stock/economy 도메인에서 Tavily 텍스트 검색만으로는 확보할 수 없는
실시간 시세·기술적 지표·거시경제 수치를 yfinance로 직접 수집한다.

이 모듈의 데이터는 research_data["market_snapshot"]에 병합되어
에이전트에게 정확한 수치를 제공한다.
"""
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _fetch_ticker(symbol: str, period: str = "3mo"):
    """yfinance 단일 종목 히스토리 수집. 실패 시 None 반환."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            logger.warning(f"{symbol}: 데이터 없음")
            return None
        return df
    except Exception as e:
        logger.error(f"{symbol} 수집 실패: {e}")
        return None


def _get_price_info(df) -> Dict:
    """종가·등락률·거래량 추출."""
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    close = float(last["Close"])
    prev_close = float(prev["Close"])
    change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
    return {
        "close": round(close, 2),
        "change_pct": round(change_pct, 2),
    }


def _calc_rsi(series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if not np.isnan(val) else 50.0


def _calc_macd(series) -> Dict:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return {
        "macd": round(float(macd.iloc[-1]), 2),
        "signal": round(float(signal.iloc[-1]), 2),
        "histogram": round(float(hist.iloc[-1]), 2),
    }


def _calc_bollinger(series, window: int = 20) -> Dict:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    current = float(series.iloc[-1])
    band_range = float(upper.iloc[-1] - lower.iloc[-1])
    position = (current - float(lower.iloc[-1])) / band_range if band_range > 0 else 0.5
    return {
        "upper": round(float(upper.iloc[-1]), 2),
        "middle": round(float(ma.iloc[-1]), 2),
        "lower": round(float(lower.iloc[-1]), 2),
        "position_pct": round(position * 100, 1),
    }


def _calc_moving_averages(series) -> Dict:
    current = float(series.iloc[-1])
    result = {}
    for p in [5, 20, 60]:
        if len(series) >= p:
            ma_val = float(series.rolling(p).mean().iloc[-1])
            result[f"ma{p}"] = round(ma_val, 2)
            result[f"ma{p}_diff_pct"] = round((current - ma_val) / ma_val * 100, 2)
    return result


# ─── 도메인별 수집 함수 ──────────────────────────────────────────────────────

def collect_stock_snapshot() -> Dict:
    """stock 도메인 정량 데이터 스냅샷.

    수집 항목:
      - KOSPI, KOSDAQ 지수 + 기술적 지표
      - USD/KRW 환율, 미 10년물 금리
      - S&P500, Nasdaq, VIX
      - 주요 종목 시가총액 상위 (삼성전자, SK하이닉스 등)
    """
    logger.info("stock 도메인 시장 데이터 수집 시작...")
    snapshot: Dict = {"collected_at": datetime.now().isoformat()}

    # 1) 주요 지수
    indices = {
        "KOSPI": "^KS11",
        "KOSDAQ": "^KQ11",
        "S&P500": "^GSPC",
        "Nasdaq": "^IXIC",
        "VIX": "^VIX",
    }
    snapshot["indices"] = {}
    kospi_df = None
    for name, symbol in indices.items():
        df = _fetch_ticker(symbol)
        info = _get_price_info(df)
        snapshot["indices"][name] = info
        if name == "KOSPI" and df is not None:
            kospi_df = df

    # 2) KOSPI 기술적 지표
    if kospi_df is not None and len(kospi_df) >= 60:
        close = kospi_df["Close"]
        bb = _calc_bollinger(close)
        macd = _calc_macd(close)
        mas = _calc_moving_averages(close)
        snapshot["kospi_technical"] = {
            "rsi_14": _calc_rsi(close),
            **macd,
            "bb_upper": bb["upper"],
            "bb_middle": bb["middle"],
            "bb_lower": bb["lower"],
            "bb_position_pct": bb["position_pct"],
            **mas,
        }

    # 3) 환율·금리
    usdkrw_df = _fetch_ticker("KRW=X", "1mo")
    us10y_df = _fetch_ticker("^TNX", "1mo")
    snapshot["macro"] = {
        "usd_krw": _get_price_info(usdkrw_df).get("close", "N/A"),
        "us_10y_yield": _get_price_info(us10y_df).get("close", "N/A"),
    }
    # 미 10년물은 1/10 단위로 반환되므로 보정
    if isinstance(snapshot["macro"]["us_10y_yield"], (int, float)):
        val = snapshot["macro"]["us_10y_yield"]
        if val > 50:  # TNX는 basis point가 아니라 10분의 1 단위
            snapshot["macro"]["us_10y_yield"] = round(val / 10, 2)

    # 4) 주요 종목
    top_stocks = {
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스",
        "373220.KS": "LG에너지솔루션",
        "035420.KS": "NAVER",
        "005380.KS": "현대차",
        "NVDA": "엔비디아",
    }
    snapshot["top_stocks"] = []
    for symbol, name in top_stocks.items():
        df = _fetch_ticker(symbol, "1mo")
        info = _get_price_info(df)
        if info:
            info["name"] = name
            info["ticker"] = symbol
            snapshot["top_stocks"].append(info)

    logger.info("stock 도메인 시장 데이터 수집 완료")
    return snapshot


def collect_economy_snapshot() -> Dict:
    """economy 도메인 정량 데이터 스냅샷."""
    logger.info("economy 도메인 시장 데이터 수집 시작...")
    snapshot: Dict = {"collected_at": datetime.now().isoformat()}

    macro_tickers = {
        "S&P500": "^GSPC",
        "DXY(달러지수)": "DX-Y.NYB",
        "미10년물": "^TNX",
        "WTI원유": "CL=F",
        "금": "GC=F",
        "USD/KRW": "KRW=X",
    }
    snapshot["macro_indicators"] = {}
    for name, symbol in macro_tickers.items():
        df = _fetch_ticker(symbol, "1mo")
        info = _get_price_info(df)
        snapshot["macro_indicators"][name] = info

    logger.info("economy 도메인 시장 데이터 수집 완료")
    return snapshot


def collect_market_snapshot(domain: str) -> Optional[Dict]:
    """도메인에 따라 적절한 시장 데이터 스냅샷을 수집.

    Args:
        domain: "stock", "economy", 등

    Returns:
        정량 데이터 dict 또는 None (지원하지 않는 도메인)
    """
    collectors = {
        "stock": collect_stock_snapshot,
        "economy": collect_economy_snapshot,
    }

    collector = collectors.get(domain)
    if collector is None:
        logger.info(f"도메인 '{domain}'은 정량 데이터 수집 대상이 아닙니다.")
        return None

    try:
        return collector()
    except Exception as e:
        logger.error(f"시장 데이터 수집 실패: {e}")
        return None
