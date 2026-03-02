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


def _calc_valuation_from_financials(symbol: str) -> Dict:
    """yfinance .financials + .balance_sheet에서 PER/PBR/EPS를 직접 계산.

    한국 주식(.KS, .KQ)은 .info에서 trailingPE/priceToBook이 N/A인 경우가 많으므로
    재무제표 데이터에서 직접 산출한다.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        price = getattr(fi, "last_price", None)
        shares = getattr(fi, "shares", None)
        if not price or not shares or shares == 0:
            return {}

        result = {}

        # Net Income → EPS → PER
        fin = ticker.financials
        if fin is not None and not fin.empty and "Net Income" in fin.index:
            net_income = fin.loc["Net Income"].iloc[0]
            if net_income and not np.isnan(net_income) and shares > 0:
                eps = net_income / shares
                result["eps"] = round(float(eps), 2)
                if eps > 0:
                    result["per_trailing"] = round(float(price / eps), 2)

        # Stockholders Equity → BPS → PBR
        bs = ticker.balance_sheet
        if bs is not None and not bs.empty:
            equity = None
            for key in ["Stockholders Equity", "Total Stockholder Equity",
                        "Total Equity Gross Minority Interest"]:
                if key in bs.index:
                    equity = bs.loc[key].iloc[0]
                    break
            if equity and not np.isnan(equity) and shares > 0:
                bps = equity / shares
                result["bps"] = round(float(bps), 2)
                if bps > 0:
                    result["pbr"] = round(float(price / bps), 2)

                # ROE = Net Income / Equity
                if "eps" in result:
                    net_income = fin.loc["Net Income"].iloc[0]
                    if equity > 0:
                        result["roe"] = round(float(net_income / equity * 100), 2)

        return result

    except Exception as e:
        logger.warning(f"{symbol} 재무제표 기반 밸류에이션 계산 실패: {e}")
        return {}


def _fetch_valuation(symbol: str) -> Dict:
    """yfinance에서 밸류에이션 지표 수집.

    전략:
      1순위: .info에서 직접 가져오기 (미국 주식에 잘 동작)
      2순위: .financials/.balance_sheet에서 직접 계산 (한국 주식 폴백)

    수집 항목: PER, PBR, 배당수익률, EPS, 시가총액, 52주 고/저, ROE
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info

        is_kr = ".KS" in symbol or ".KQ" in symbol
        currency = "KRW" if is_kr else info.get("currency", "USD")

        if not info or info.get("regularMarketPrice") is None:
            # info 자체가 없으면 fast_info + financials 폴백
            result = _calc_valuation_from_financials(symbol)
            fi = ticker.fast_info
            result["market_cap"] = _format_market_cap(getattr(fi, "market_cap", None), currency)
            result.setdefault("per_trailing", "N/A")
            result.setdefault("per_forward", "N/A")
            result.setdefault("pbr", "N/A")
            result.setdefault("dividend_yield", "N/A")
            result.setdefault("eps", "N/A")
            return result

        result = {}

        # PER
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        result["per_trailing"] = round(float(trailing_pe), 2) if trailing_pe else "N/A"
        result["per_forward"] = round(float(forward_pe), 2) if forward_pe else "N/A"

        # PBR
        pbr = info.get("priceToBook")
        result["pbr"] = round(float(pbr), 2) if pbr else "N/A"

        # 배당수익률 — dividendRate / price로 직접 계산 (가장 정확)
        # yfinance의 dividendYield는 통화/지역에 따라 형식이 불일치하므로 폴백만 사용
        div_rate = info.get("dividendRate")
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if div_rate and price and price > 0:
            result["dividend_yield"] = round(float(div_rate) / float(price) * 100, 2)
        else:
            div_yield = info.get("dividendYield")
            if div_yield:
                div_val = float(div_yield)
                # 20% 초과는 데이터 오류로 판단
                if div_val > 20:
                    result["dividend_yield"] = "N/A"
                elif div_val > 1:
                    result["dividend_yield"] = round(div_val, 2)
                else:
                    result["dividend_yield"] = round(div_val * 100, 2)
            else:
                result["dividend_yield"] = "N/A"

        # EPS
        eps = info.get("trailingEps")
        result["eps"] = round(float(eps), 2) if eps else "N/A"

        # 시가총액
        mcap = info.get("marketCap")
        result["market_cap"] = _format_market_cap(mcap, currency)

        # 52주 고/저
        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")
        result["week52_high"] = round(float(high_52w), 2) if high_52w else "N/A"
        result["week52_low"] = round(float(low_52w), 2) if low_52w else "N/A"

        # ROE
        roe = info.get("returnOnEquity")
        result["roe"] = round(float(roe) * 100, 2) if roe else "N/A"

        # 한국 주식 폴백: .info에서 N/A인 핵심 지표를 재무제표로 보완
        if is_kr:
            needs_fallback = (
                result["per_trailing"] == "N/A" or
                result["pbr"] == "N/A" or
                result["eps"] == "N/A"
            )
            if needs_fallback:
                calc = _calc_valuation_from_financials(symbol)
                if calc:
                    if result["per_trailing"] == "N/A" and "per_trailing" in calc:
                        result["per_trailing"] = calc["per_trailing"]
                    if result["pbr"] == "N/A" and "pbr" in calc:
                        result["pbr"] = calc["pbr"]
                    if result["eps"] == "N/A" and "eps" in calc:
                        result["eps"] = calc["eps"]
                    if result["roe"] == "N/A" and "roe" in calc:
                        result["roe"] = calc["roe"]

        return result

    except Exception as e:
        logger.warning(f"{symbol} 밸류에이션 수집 실패: {e}")
        return {}


def _format_market_cap(mcap, currency: str = "USD") -> str:
    """시가총액을 통화에 맞게 사람이 읽기 쉬운 형태로 변환."""
    if mcap is None:
        return "N/A"
    mcap = float(mcap)

    if currency == "KRW":
        # 한국 원화: 조원/억원 단위
        if mcap >= 1e12:
            return f"{mcap / 1e12:.1f}조원"
        elif mcap >= 1e8:
            return f"{mcap / 1e8:.0f}억원"
        return f"{mcap:,.0f}원"
    else:
        # USD 등 외화: T/B/M 단위
        if mcap >= 1e12:
            return f"${mcap / 1e12:.2f}T"
        elif mcap >= 1e9:
            return f"${mcap / 1e9:.1f}B"
        elif mcap >= 1e6:
            return f"${mcap / 1e6:.0f}M"
        return f"${mcap:,.0f}"


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

    # 4) 주요 종목 — 가격 + 밸류에이션 지표
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
            # 밸류에이션 지표 추가 (PER, PBR, 배당수익률, EPS 등)
            valuation = _fetch_valuation(symbol)
            if valuation:
                info["valuation"] = valuation
            snapshot["top_stocks"].append(info)

    # 5) 시장 밸류에이션 요약 — 개별 종목 PER/PBR 중위값으로 산출
    valid_pers = [s["valuation"]["per_trailing"] for s in snapshot["top_stocks"]
                  if s.get("valuation", {}).get("per_trailing") not in ("N/A", None)
                  and isinstance(s["valuation"]["per_trailing"], (int, float))
                  and s["valuation"]["per_trailing"] > 0]
    valid_pbrs = [s["valuation"]["pbr"] for s in snapshot["top_stocks"]
                  if s.get("valuation", {}).get("pbr") not in ("N/A", None)
                  and isinstance(s["valuation"]["pbr"], (int, float))
                  and s["valuation"]["pbr"] > 0]
    if valid_pers or valid_pbrs:
        snapshot["market_valuation"] = {}
        if valid_pers:
            snapshot["market_valuation"]["top_stocks_median_per"] = round(float(np.median(valid_pers)), 2)
        if valid_pbrs:
            snapshot["market_valuation"]["top_stocks_median_pbr"] = round(float(np.median(valid_pbrs)), 2)

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

    # S&P500 밸류에이션 (글로벌 시장 과열/저평가 판단용)
    sp500_val = _fetch_valuation("^GSPC")
    if not sp500_val or sp500_val.get("per_trailing") == "N/A":
        sp500_val = _fetch_valuation("SPY")  # ETF 폴백
    if sp500_val:
        snapshot["market_valuation"] = {
            "sp500_per": sp500_val.get("per_trailing", "N/A"),
            "sp500_pbr": sp500_val.get("pbr", "N/A"),
            "sp500_dividend_yield": sp500_val.get("dividend_yield", "N/A"),
        }

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
