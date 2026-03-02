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
        lines = [f"주제: {research_data.get('topic', '알 수 없음')}"]

        # AI 요약
        ai_summary = research_data.get("ai_summary", "")
        if ai_summary:
            lines.append(f"\n[AI 사전 요약]\n{ai_summary[:600]}")

        # 메인 소스
        sources = research_data.get("sources", [])
        if sources:
            lines.append(f"\n[주요 소스 ({len(sources)}건)]")
            for i, s in enumerate(sources[:8], 1):
                lines.append(f"\n--- 소스 {i}: {s.get('title', '')} ---")
                lines.append(s.get("content", "")[:400])

        # 반대 의견 소스
        counter = research_data.get("counter_sources", [])
        if counter:
            lines.append(f"\n[반대 의견/리스크 소스 ({len(counter)}건)]")
            for i, s in enumerate(counter[:4], 1):
                lines.append(f"\n--- 반론 소스 {i}: {s.get('title', '')} ---")
                lines.append(s.get("content", "")[:300])

        lines.append(f"\n수집 시각: {research_data.get('collected_at', '')}")
        return "\n".join(lines)

    # ── 서브클래스에서 구현 ──────────────────────────────────────────────────

    def analyze(self, research_data: dict) -> AgentReport:
        raise NotImplementedError

    def critique(self, other_report: AgentReport, research_data: dict) -> AgentCritique:
        raise NotImplementedError

    def review(self, author_report: AgentReport, research_data: dict) -> ReviewerReport:
        """MARS Phase 2: Author의 분석을 전문 관점에서 독립 리뷰."""
        raise NotImplementedError
