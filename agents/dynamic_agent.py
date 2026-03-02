"""
dynamic_agent.py — 도메인 프리셋 기반 동적 에이전트

도메인별 에이전트 설정(domains.py)을 받아 런타임에 페르소나를 구성한다.
개별 에이전트 파일을 도메인마다 만들 필요 없이, 프리셋만으로 에이전트가 동작한다.
"""
import logging
from typing import List

import anthropic

from .base_agent import BaseAgent, AgentReport, AgentCritique

logger = logging.getLogger(__name__)


def _build_system_prompt(agent_cfg: dict, stance_values: List[str]) -> str:
    """에이전트 설정 dict로부터 system prompt를 생성한다."""
    stances_str = " / ".join(stance_values)
    return f"""당신은 '{agent_cfg["name"]}'이라는 이름의 전문 분석가입니다.

[역할] {agent_cfg["role"]}

[분석 관점]
{agent_cfg["perspective"]}

[신뢰하는 근거]
{agent_cfg["trust"]}

[불신하는 근거]
{agent_cfg["distrust"]}

[반론 스타일]
{agent_cfg["critique_style"]}

[출력 규칙]
- 한국어로 분석
- 반드시 JSON 형식으로만 응답
- stance는 반드시 {stances_str} 중 하나
- confidence_score는 0~100 사이 정수
"""


class DynamicAgent(BaseAgent):
    """도메인 프리셋 설정을 받아 동적으로 구성되는 에이전트."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        agent_cfg: dict,
        stance_values: List[str],
    ):
        super().__init__(client, model)
        self.name = agent_cfg["name"]
        self.role = agent_cfg["role"]
        self.avatar = agent_cfg["avatar"]
        self.stance_values = stance_values
        self.system_prompt = _build_system_prompt(agent_cfg, stance_values)

    def analyze(self, research_data: dict) -> AgentReport:
        """Phase 1: 독립 분석."""
        data_text = self._format_research_data(research_data)
        stances_str = " / ".join(self.stance_values)

        prompt = f"""아래 리서치 데이터를 당신의 전문 관점에서 분석하세요.

{data_text}

[분석 가이드]
- 당신의 전문 관점에서 가장 중요한 시그널은 무엇인가?
- 어떤 근거가 당신의 판단을 지지하고, 어떤 부분이 불확실한가?
- 300자 이상 깊이 있는 분석을 제공하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "300자 이상 분석 (구체적 근거 포함)",
  "key_points": ["핵심 인사이트 1", "핵심 인사이트 2", "핵심 인사이트 3"],
  "confidence_score": 75,
  "stance": "{self.stance_values[0]}"
}}

stance: {stances_str} 중 하나
confidence_score: 0~100"""

        result = self._call_llm([{"role": "user", "content": prompt}])
        data = self._parse_json_response(result)

        # stance 유효성 검증
        raw_stance = data.get("stance", self.stance_values[len(self.stance_values) // 2]).upper()
        if raw_stance not in self.stance_values:
            raw_stance = self.stance_values[len(self.stance_values) // 2]  # 중립 폴백

        return AgentReport(
            agent_name=self.name,
            role=self.role,
            avatar=self.avatar,
            analysis=data.get("analysis", result[:600]),
            key_points=data.get("key_points", ["분석 완료"]),
            confidence_score=max(0, min(100, int(data.get("confidence_score", 50)))),
            stance=raw_stance,
        )

    def critique(self, other_report: AgentReport, research_data: dict) -> AgentCritique:
        """Phase 2: 교차 반론."""
        prompt = f"""{self.name}의 관점에서 아래 분석에 날카로운 반론을 제시하세요.

[{other_report.agent_name} — {other_report.role}의 분석]
의견: {other_report.stance} (확신도: {other_report.confidence_score})
주장: {other_report.analysis[:400]}
핵심: {', '.join(other_report.key_points[:3])}

[반론 가이드]
- 상대 주장의 근본 가정이나 데이터 취약점을 공격하세요.
- 당신의 전문 관점에서 상대가 간과한 핵심 요소를 지적하세요.
- 150~250자, 감정적 표현 없이 논리적으로 반박하세요."""

        result = self._call_llm([{"role": "user", "content": prompt}])
        return AgentCritique(
            from_agent=self.name,
            to_agent=other_report.agent_name,
            critique=result.strip()[:500],
        )
