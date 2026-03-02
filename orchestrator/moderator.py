"""
moderator.py — Phase 3: 하이브리드 종합 판단

두 단계로 최종 판단을 내린다:
  1. 규칙 기반 선행 집계: confidence_score 가중 투표
  2. LLM 품질 평가: 토론 내용의 논리적 강점/약점 기반 최종 판단

도메인별 stance 값과 스코어 매핑을 외부에서 주입받는다.
"""
import json
import logging
import re
from typing import Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)


class Moderator:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        stance_score: Dict[str, float],
        stance_thresholds: Dict[str, float],
        stance_values: List[str],
    ):
        self.client = client
        self.model = model
        self.stance_score = stance_score
        self.stance_thresholds = stance_thresholds
        self.stance_values = stance_values

    def synthesize(
        self,
        reports: List[dict],
        critiques: List[dict],
        research_data: dict,
    ) -> dict:
        """
        Phase 3: Moderator 종합 판단.

        Args:
            reports: Phase 1 보고서 (dict 리스트)
            critiques: Phase 2 반론 (dict 리스트)
            research_data: 원본 리서치 데이터

        Returns:
            verdict dict
        """
        # 1) 규칙 기반 집계
        weighted_score, avg_confidence = self._weighted_vote(reports)
        rule_stance = self._score_to_stance(weighted_score)
        logger.info(f"규칙 기반 판단: {rule_stance} (가중 점수: {weighted_score:.2f})")

        # 2) LLM 종합 판단
        debate_text = self._format_debate(reports, critiques)
        stances_str = " / ".join(self.stance_values)

        prompt = f"""당신은 멀티에이전트 토론의 중재자입니다.
아래 에이전트들의 토론 내용을 종합하여 최종 판단을 내려주세요.

[리서치 주제]
{research_data.get("topic", "알 수 없음")}

[토론 내용]
{debate_text}

[규칙 기반 선행 판단: {rule_stance}]

종합 시 다음을 고려하세요:
1. 각 에이전트의 논리적 강점과 약점
2. 데이터 근거의 신뢰도
3. 반론에서 효과적으로 반박된 주장과 그렇지 못한 주장
4. 전체적인 합의 방향과 핵심 쟁점

반드시 아래 JSON으로만 응답:
{{
  "final_stance": "{self.stance_values[0]}",
  "confidence_score": 68,
  "summary": "종합 근거 설명 (200자 이상)",
  "key_insights": ["핵심 인사이트 1", "핵심 인사이트 2", "핵심 인사이트 3"],
  "risk_factors": ["리스크 1", "리스크 2"],
  "action_items": ["실행 제안 1", "실행 제안 2"],
  "debate_quality": "토론 품질 한줄 평가"
}}

stance: {stances_str} 중 하나"""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system="당신은 멀티에이전트 토론 중재자입니다. 논리의 질과 데이터 신뢰성을 기준으로 공정하게 판단합니다. JSON만 반환하세요.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = re.sub(r"```(?:json)?\s*|```\s*", "", resp.content[0].text).strip()
            match = re.search(r"\{[\s\S]*\}", text)
            result = json.loads(match.group()) if match else {}
        except Exception as e:
            logger.error(f"Moderator LLM 호출 실패: {e}")
            result = {
                "final_stance": rule_stance,
                "confidence_score": 50,
                "summary": "에이전트 토론을 종합한 결과입니다.",
            }

        # stance 유효성 검증
        final_stance = result.get("final_stance", rule_stance)
        if final_stance not in self.stance_values:
            final_stance = rule_stance

        return {
            "final_stance": final_stance,
            "confidence_score": max(0, min(100, int(result.get("confidence_score", int(avg_confidence))))),
            "summary": result.get("summary", ""),
            "key_insights": result.get("key_insights", []),
            "risk_factors": result.get("risk_factors", []),
            "action_items": result.get("action_items", []),
            "debate_quality": result.get("debate_quality", ""),
            "stance_votes": {r["agent_name"]: r["stance"] for r in reports},
            "rule_based_stance": rule_stance,
        }

    def _weighted_vote(self, reports: List[dict]) -> tuple:
        """confidence_score를 가중치로 stance 투표."""
        total_w, weighted_sum, conf_sum = 0, 0, 0
        for r in reports:
            w = r.get("confidence_score", 50)
            stance = r.get("stance", "NEUTRAL")
            s = self.stance_score.get(stance, 0)
            weighted_sum += s * w
            total_w += w
            conf_sum += w
        if total_w == 0:
            return 0.0, 50.0
        return weighted_sum / total_w, conf_sum / len(reports)

    def _score_to_stance(self, score: float) -> str:
        """가중 점수를 stance로 변환."""
        pos = self.stance_thresholds.get("positive", 0.35)
        neg = self.stance_thresholds.get("negative", -0.35)
        if score > pos:
            return self.stance_values[0]   # 가장 긍정적
        if score < neg:
            return self.stance_values[-1]  # 가장 부정적
        # 중간값
        mid = len(self.stance_values) // 2
        return self.stance_values[mid]

    def _format_debate(self, reports: List[dict], critiques: List[dict]) -> str:
        """토론 내용을 텍스트로 정리."""
        lines = ["[Phase 1 — 개별 분석]"]
        for r in reports:
            lines.append(
                f"\n{r.get('avatar','')} {r['agent_name']} ({r['role']})"
                f"\n  판단: {r['stance']} (확신도 {r['confidence_score']}%)"
                f"\n  분석: {r['analysis'][:300]}"
                f"\n  핵심: {', '.join(r.get('key_points', [])[:3])}"
            )

        lines.append("\n\n[Phase 2 — 교차 반론]")
        for c in critiques:
            lines.append(
                f"\n{c['from_agent']} → {c['to_agent']}: {c['critique'][:250]}"
            )

        return "\n".join(lines)
