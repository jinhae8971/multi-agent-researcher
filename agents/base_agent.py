"""
base_agent.py — 모든 에이전트의 공통 인터페이스

Korean Stock Agent의 검증된 패턴을 기반으로 범용화.
핵심 데이터클래스(AgentReport, AgentCritique, ReviewerReport)와
LLM 호출/JSON 파싱 유틸리티를 제공한다.

MARS(Multi-Agent Review System) 패턴 지원:
  - ReviewerReport: Reviewer 에이전트의 리뷰 결과
  - review() 메서드: Author 분석에 대한 독립 리뷰
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import List

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class AgentReport:
    agent_name: str
    role: str
    avatar: str
    analysis: str
    key_points: List[str]
    confidence_score: int   # 0~100
    stance: str             # 도메인별 커스텀 (BUY/HOLD/SELL, BULLISH/NEUTRAL/BEARISH 등)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "avatar": self.avatar,
            "analysis": self.analysis,
            "key_points": self.key_points,
            "confidence_score": self.confidence_score,
            "stance": self.stance,
        }


@dataclass
class AgentCritique:
    from_agent: str
    to_agent: str
    critique: str

    def to_dict(self) -> dict:
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "critique": self.critique,
        }


@dataclass
class ReviewerReport:
    """MARS 패턴: Reviewer 에이전트의 독립 리뷰 결과."""
    reviewer_name: str          # 리뷰어 이름
    role: str                   # 전문 역할
    avatar: str                 # 이모지
    decision: str               # "agree" | "disagree" | "partial"
    confidence: int             # 1~5 (MARS 논문 기준)
    justification: str          # 판단 근거 (200~400자)
    suggested_revision: str     # 수정 제안 (disagree/partial 시)
    key_concerns: List[str]     # 핵심 우려사항 (최대 3개)

    def to_dict(self) -> dict:
        return {
            "reviewer_name": self.reviewer_name,
            "role": self.role,
            "avatar": self.avatar,
            "decision": self.decision,
            "confidence": self.confidence,
            "justification": self.justification,
            "suggested_revision": self.suggested_revision,
            "key_concerns": self.key_concerns,
        }


class BaseAgent:
    name: str = ""
    role: str = ""
    avatar: str = ""
    system_prompt: str = ""

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model

    # ── LLM 호출 ────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list, max_tokens: int = 2048) -> str:
        """LLM 호출. 예외 시 에러 메시지 반환."""
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            logger.error(f"[{self.name}] LLM 호출 실패: {e}")
            raise

    def _parse_json_response(self, text: str) -> dict:
        """LLM 응답에서 JSON 추출. 코드블록·설명문 혼재에도 안전."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                # 줄바꿈/탭 정제 후 재시도
                cleaned = match.group().replace("\n", " ").replace("\t", " ")
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
        logger.warning(f"[{self.name}] JSON 파싱 실패, 빈 dict 반환")
        return {}

    def _format_research_data(self, research_data: dict) -> str:
        """웹 리서치 데이터를 에이전트가 읽을 수 있는 텍스트로 변환."""
        collected_at = research_data.get("collected_at", "")

        lines = [
            f"주제: {research_data.get('topic', '알 수 없음')}",
            f"데이터 수집 시각: {collected_at}",
            "",
            "⚠️ [데이터 정합성 필수 규칙 — 반드시 준수]",
            "1. 아래 소스 데이터에 명시된 수치·팩트만 사용하세요.",
            "2. 당신의 학습 데이터(training data)에 기반한 과거 수치를 절대 사용하지 마세요.",
            "3. 특히 주가지수, 환율, 금리 등 시계열 수치는 소스에서 직접 인용한 것만 사용하세요.",
            "4. 소스에 명시되지 않은 구체적 수치(예: 지수 포인트, %)는 추측하지 말고 '소스 미확인'으로 표시하세요.",
            "5. 소스 내 날짜를 확인하여 오래된 데이터인지 최신 데이터인지 구분하세요.",
            "",
        ]

        # 📊 실시간 시장 정량 데이터 (yfinance) — 가장 신뢰할 수 있는 수치
        market = research_data.get("market_snapshot")
        if market:
            lines.append(self._format_market_snapshot(market))

        # AI 요약
        ai_summary = research_data.get("ai_summary", "")
        if ai_summary:
            lines.append(f"[AI 사전 요약]\n{ai_summary[:600]}")

        # 메인 소스
        sources = research_data.get("sources", [])
        if sources:
            lines.append(f"\n[주요 소스 ({len(sources)}건)]")
            for i, s in enumerate(sources[:10], 1):
                lines.append(f"\n--- 소스 {i}: {s.get('title', '')} ---")
                lines.append(f"URL: {s.get('url', '')}")
                lines.append(s.get("content", "")[:800])

        # 보강 소스 (정량 데이터 — 시세, 환율, 금리 등)
        supplementary = research_data.get("supplementary_sources", [])
        if supplementary:
            lines.append(f"\n[📊 정량 보강 데이터 ({len(supplementary)}건) — 수치 인용 시 이 소스를 우선 참조]")
            for i, s in enumerate(supplementary[:6], 1):
                lines.append(f"\n--- 보강 소스 {i}: {s.get('title', '')} ---")
                lines.append(f"URL: {s.get('url', '')}")
                lines.append(s.get("content", "")[:800])

        # 반대 의견 소스
        counter = research_data.get("counter_sources", [])
        if counter:
            lines.append(f"\n[반대 의견/리스크 소스 ({len(counter)}건)]")
            for i, s in enumerate(counter[:4], 1):
                lines.append(f"\n--- 반론 소스 {i}: {s.get('title', '')} ---")
                lines.append(f"URL: {s.get('url', '')}")
                lines.append(s.get("content", "")[:400])

        lines.append(f"\n[재확인] 데이터 수집 시각: {collected_at}")
        lines.append("위 소스에 포함된 구체적 수치만 인용하세요. 학습 데이터의 과거 수치를 사용하면 심각한 오류가 됩니다.")
        return "\n".join(lines)

    @staticmethod
    def _format_market_snapshot(market: dict) -> str:
        """yfinance 시장 스냅샷을 에이전트용 텍스트로 변환."""
        lines = [
            "",
            "=" * 50,
            "📊 [실시간 시장 데이터 — yfinance 직접 수집, 가장 정확한 수치]",
            f"수집 시각: {market.get('collected_at', 'N/A')}",
            "⚠️ 지수·환율·금리 인용 시 반드시 이 섹션의 수치를 사용하세요.",
            "=" * 50,
        ]

        # 지수
        indices = market.get("indices", {})
        if indices:
            lines.append("\n=== 주요 지수 ===")
            for name, data in indices.items():
                if isinstance(data, dict) and data.get("close"):
                    lines.append(
                        f"  {name:10s}: {data['close']:>12,.2f}  ({data.get('change_pct', 0):+.2f}%)"
                    )

        # 기술적 지표
        tech = market.get("kospi_technical", {})
        if tech:
            lines.append("\n=== KOSPI 기술적 지표 ===")
            lines.append(f"  RSI(14)     : {tech.get('rsi_14', 'N/A')}")
            lines.append(
                f"  MACD        : {tech.get('macd', 'N/A')}  "
                f"Signal: {tech.get('signal', 'N/A')}  "
                f"Hist: {tech.get('histogram', 'N/A')}"
            )
            lines.append(f"  볼린저밴드   : 밴드 내 위치 {tech.get('bb_position_pct', 'N/A')}%")
            lines.append(
                f"  이동평균     : MA5={tech.get('ma5', 'N/A')} "
                f"MA20={tech.get('ma20', 'N/A')} "
                f"MA60={tech.get('ma60', 'N/A')}"
            )

        # 거시
        macro = market.get("macro", {})
        if macro:
            lines.append("\n=== 환율·금리 ===")
            lines.append(f"  USD/KRW     : {macro.get('usd_krw', 'N/A')}")
            lines.append(f"  미 10년물   : {macro.get('us_10y_yield', 'N/A')}%")

        macro_ind = market.get("macro_indicators", {})
        if macro_ind:
            lines.append("\n=== 거시경제 지표 ===")
            for name, data in macro_ind.items():
                if isinstance(data, dict) and data.get("close"):
                    lines.append(
                        f"  {name:14s}: {data['close']:>12,.2f}  ({data.get('change_pct', 0):+.2f}%)"
                    )

        # 주요 종목
        stocks = market.get("top_stocks", [])
        if stocks:
            lines.append("\n=== 주요 종목 ===")
            for s in stocks:
                lines.append(
                    f"  {s.get('name', ''):10s}: {s.get('close', 0):>12,.2f}  "
                    f"({s.get('change_pct', 0):+.2f}%)"
                )

        lines.append("=" * 50)
        return "\n".join(lines)

    # ── 서브클래스에서 구현 ──────────────────────────────────────────────────

    def analyze(self, research_data: dict) -> AgentReport:
        raise NotImplementedError

    def critique(self, other_report: AgentReport, research_data: dict) -> AgentCritique:
        raise NotImplementedError

    def review(self, author_report: AgentReport, research_data: dict) -> ReviewerReport:
        """MARS Phase 2: Author의 분석을 전문 관점에서 독립 리뷰."""
        raise NotImplementedError
