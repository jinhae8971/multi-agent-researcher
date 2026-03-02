"""
dynamic_agent.py — 도메인 프리셋 기반 동적 에이전트

도메인별 에이전트 설정(domains.py)을 받아 런타임에 페르소나를 구성한다.
개별 에이전트 파일을 도메인마다 만들 필요 없이, 프리셋만으로 에이전트가 동작한다.

MARS 패턴 지원:
  - role_type="author": Author 에이전트 (종합 분석 생성)
  - role_type="reviewer": Reviewer 에이전트 (독립 리뷰)
  - role_type="analyst": 기존 토론 모드 (Phase 1+2)
"""
import logging
from typing import List

import anthropic

from .base_agent import BaseAgent, AgentReport, AgentCritique, ReviewerReport

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


def _build_author_system_prompt(agent_cfg: dict, stance_values: List[str]) -> str:
    """MARS Author 에이전트용 system prompt 생성."""
    stances_str = " / ".join(stance_values)
    return f"""당신은 '{agent_cfg["name"]}'이라는 이름의 수석 리서치 분석가입니다.

[역할] {agent_cfg["role"]}

[분석 관점]
{agent_cfg["perspective"]}

[분석 원칙]
- Chain-of-Thought: 추론 과정을 단계별로 명시적으로 보여주세요.
- 다각적 관점: 찬성/반대/중립 논거를 모두 포함하세요.
- 데이터 기반: 구체적 수치와 출처를 인용하세요.
- 불확실성 인정: 확실하지 않은 부분은 명시하세요.

[출력 규칙]
- 한국어로 분석
- 반드시 JSON 형식으로만 응답
- stance는 반드시 {stances_str} 중 하나
- confidence_score는 0~100 사이 정수
"""


def _build_reviewer_system_prompt(agent_cfg: dict, stance_values: List[str]) -> str:
    """MARS Reviewer 에이전트용 system prompt 생성."""
    return f"""당신은 '{agent_cfg["name"]}'이라는 이름의 전문 리뷰어입니다.

[역할] {agent_cfg["role"]}

[리뷰 관점]
{agent_cfg["perspective"]}

[신뢰하는 근거]
{agent_cfg["trust"]}

[불신하는 근거]
{agent_cfg["distrust"]}

[리뷰 원칙]
- Author의 분석을 당신의 전문 관점에서 독립적으로 평가합니다.
- 다른 Reviewer의 의견은 고려하지 않습니다 (독립 리뷰).
- 논리의 강점과 약점을 모두 평가합니다.
- 구체적인 수정 제안을 포함합니다.

[출력 규칙]
- 한국어로 리뷰
- 반드시 JSON 형식으로만 응답
"""


class DynamicAgent(BaseAgent):
    """도메인 프리셋 설정을 받아 동적으로 구성되는 에이전트."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        agent_cfg: dict,
        stance_values: List[str],
        role_type: str = "analyst",  # "analyst" | "author" | "reviewer"
    ):
        super().__init__(client, model)
        self.name = agent_cfg["name"]
        self.role = agent_cfg["role"]
        self.avatar = agent_cfg["avatar"]
        self.stance_values = stance_values
        self.role_type = role_type

        # 역할에 따라 system prompt 분기
        if role_type == "author":
            self.system_prompt = _build_author_system_prompt(agent_cfg, stance_values)
        elif role_type == "reviewer":
            self.system_prompt = _build_reviewer_system_prompt(agent_cfg, stance_values)
        else:
            self.system_prompt = _build_system_prompt(agent_cfg, stance_values)

    def analyze(self, research_data: dict) -> AgentReport:
        """Phase 1: 독립 분석. Author 모드일 경우 CoT 기반 종합 분석 생성."""
        data_text = self._format_research_data(research_data)
        stances_str = " / ".join(self.stance_values)

        if self.role_type == "author":
            prompt = f"""아래 리서치 데이터를 종합적으로 분석하세요.

{data_text}

[분석 가이드 — Author 모드]
1. 먼저 데이터의 핵심 팩트를 정리하세요.
2. 찬성(긍정적) 근거와 반대(부정적) 근거를 각각 나열하세요.
3. 불확실한 영역과 추가 검증이 필요한 부분을 명시하세요.
4. 위 분석을 종합하여 최종 판단을 내리세요.
5. 500자 이상 깊이 있는 분석을 제공하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "500자 이상 종합 분석 (추론 과정 포함, 찬반 근거 모두 포함)",
  "key_points": ["핵심 인사이트 1", "핵심 인사이트 2", "핵심 인사이트 3"],
  "confidence_score": 75,
  "stance": "{self.stance_values[0]}"
}}

stance: {stances_str} 중 하나
confidence_score: 0~100"""
        else:
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

        max_tokens = 3072 if self.role_type == "author" else 2048
        result = self._call_llm([{"role": "user", "content": prompt}], max_tokens=max_tokens)
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
        """Phase 2: 교차 반론 (기존 토론 모드 전용)."""
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

    def review(self, author_report: AgentReport, research_data: dict) -> ReviewerReport:
        """MARS Step 2: Author의 분석을 전문 관점에서 독립 리뷰.

        다른 Reviewer의 의견은 보지 않으며, 오직 Author의 분석만 평가한다.
        이것이 MARS의 핵심 — Reviewer 간 상호작용 제거로 토큰 50% 절감.
        """
        data_text = self._format_research_data(research_data)

        prompt = f"""당신의 전문 관점에서 아래 Author의 분석을 독립적으로 리뷰하세요.

[리서치 원본 데이터 요약]
{data_text[:1500]}

[Author의 분석]
판단: {author_report.stance} (확신도: {author_report.confidence_score}%)
분석: {author_report.analysis}
핵심 인사이트: {', '.join(author_report.key_points[:5])}

[리뷰 가이드]
1. Author의 분석이 당신의 전문 분야에서 타당한가?
2. Author가 놓친 중요한 관점이나 데이터가 있는가?
3. 분석의 논리적 강점과 약점은 무엇인가?
4. 수정이 필요하다면 구체적으로 어떤 부분인가?

반드시 아래 JSON으로만 응답:
{{
  "decision": "agree",
  "confidence": 4,
  "justification": "200~400자 판단 근거",
  "suggested_revision": "수정 제안 (agree 시 빈 문자열 가능)",
  "key_concerns": ["우려사항 1", "우려사항 2"]
}}

decision: agree / disagree / partial 중 하나
confidence: 1(매우 낮음) ~ 5(매우 높음)"""

        result = self._call_llm([{"role": "user", "content": prompt}])
        data = self._parse_json_response(result)

        # decision 유효성 검증
        decision = data.get("decision", "partial").lower()
        if decision not in ("agree", "disagree", "partial"):
            decision = "partial"

        # confidence 유효성 검증 (1~5)
        confidence = max(1, min(5, int(data.get("confidence", 3))))

        return ReviewerReport(
            reviewer_name=self.name,
            role=self.role,
            avatar=self.avatar,
            decision=decision,
            confidence=confidence,
            justification=data.get("justification", result[:400]),
            suggested_revision=data.get("suggested_revision", ""),
            key_concerns=data.get("key_concerns", []),
        )
