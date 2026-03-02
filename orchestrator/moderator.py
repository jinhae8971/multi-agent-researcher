"""
moderator.py — Phase 3: 하이브리드 종합 판단

두 가지 모드를 지원한다:

[기존 Debate 모드] synthesize()
  1. 규칙 기반 선행 집계: confidence_score 가중 투표
  2. LLM 품질 평가: 토론 내용의 논리적 강점/약점 기반 최종 판단

[MARS 모드] meta_review()
  1. Reviewer 리뷰 결과 규칙 기반 집계
  2. LLM Meta-Reviewer: Author 분석 + 리뷰 통합 → 최종 판단

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

    # ══════════════════════════════════════════════════════════════════════════
    # MARS 모드: Meta-Reviewer 기반 종합 판단
    # ══════════════════════════════════════════════════════════════════════════

    def meta_review(
        self,
        author_report: dict,
        reviews: List[dict],
        research_data: dict,
        rebuttal_report: Optional[dict] = None,
    ) -> dict:
        """MARS Meta-Reviewer: Author 분석 + Reviewer 리뷰를 통합하여 최종 판단.

        Args:
            author_report: Author의 분석 (AgentReport.to_dict())
            reviews: Reviewer들의 리뷰 (ReviewerReport.to_dict() 리스트)
            research_data: 원본 리서치 데이터
            rebuttal_report: Rebuttal 결과 (선택, AgentReport.to_dict())

        Returns:
            verdict dict
        """
        # 1) 규칙 기반 리뷰 집계
        review_score, avg_confidence = self._aggregate_reviews(reviews)
        rule_stance = self._review_score_to_stance(review_score, author_report)
        logger.info(
            f"MARS 규칙 기반 판단: {rule_stance} "
            f"(리뷰 점수: {review_score:.2f}, 평균 신뢰도: {avg_confidence:.1f}/5)"
        )

        # 2) LLM Meta-Reviewer 종합 판단
        mars_text = self._format_mars_debate(author_report, reviews, rebuttal_report)
        stances_str = " / ".join(self.stance_values)

        # 최종 분석은 rebuttal이 있으면 rebuttal, 없으면 author 원본
        final_author = rebuttal_report if rebuttal_report else author_report

        prompt = f"""당신은 MARS(Multi-Agent Review System)의 Meta-Reviewer입니다.
Author의 분석과 전문가 Reviewer들의 리뷰를 종합하여 최종 판단을 내려주세요.

[리서치 주제]
{research_data.get("topic", "알 수 없음")}

{mars_text}

[규칙 기반 선행 판단: {rule_stance}]

[Meta-Reviewer 판단 기준]
1. Author의 분석 논리와 데이터 근거의 타당성
2. Reviewer들이 공통으로 지적한 핵심 약점
3. Reviewer들이 인정한 분석의 강점
4. 리뷰어 간 의견이 분분한 쟁점 사항
5. 당신 자신의 독립적인 판단 (리뷰어만 의존하지 말 것)

반드시 아래 JSON으로만 응답:
{{
  "final_stance": "{self.stance_values[0]}",
  "confidence_score": 68,
  "summary": "Meta-Reviewer 종합 근거 설명 (300자 이상)",
  "key_insights": ["핵심 인사이트 1", "핵심 인사이트 2", "핵심 인사이트 3"],
  "risk_factors": ["리스크 1", "리스크 2"],
  "action_items": ["실행 제안 1", "실행 제안 2"],
  "review_summary": "리뷰 종합 한줄 평가",
  "debate_quality": "분석-리뷰 품질 한줄 평가"
}}

stance: {stances_str} 중 하나"""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=(
                    "당신은 MARS Meta-Reviewer입니다. "
                    "Author의 분석과 Reviewer들의 독립 리뷰를 종합하여 "
                    "공정하고 균형 잡힌 최종 판단을 내립니다. "
                    "Reviewer 의견만 의존하지 말고, 당신 자신의 판단도 반영하세요. "
                    "JSON만 반환하세요."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            text = re.sub(r"```(?:json)?\s*|```\s*", "", resp.content[0].text).strip()
            match = re.search(r"\{[\s\S]*\}", text)
            result = json.loads(match.group()) if match else {}
        except Exception as e:
            logger.error(f"Meta-Reviewer LLM 호출 실패: {e}")
            result = {
                "final_stance": rule_stance,
                "confidence_score": 50,
                "summary": "Meta-Reviewer 자동 종합 결과입니다.",
            }

        # stance 유효성 검증
        final_stance = result.get("final_stance", rule_stance)
        if final_stance not in self.stance_values:
            final_stance = rule_stance

        return {
            "final_stance": final_stance,
            "confidence_score": max(0, min(100, int(result.get("confidence_score", int(avg_confidence * 20))))),
            "summary": result.get("summary", ""),
            "key_insights": result.get("key_insights", []),
            "risk_factors": result.get("risk_factors", []),
            "action_items": result.get("action_items", []),
            "review_summary": result.get("review_summary", ""),
            "debate_quality": result.get("debate_quality", ""),
            "author_stance": final_author.get("stance", "NEUTRAL"),
            "author_confidence": final_author.get("confidence_score", 50),
            "reviewer_decisions": {r["reviewer_name"]: r["decision"] for r in reviews},
            "rule_based_stance": rule_stance,
            "had_rebuttal": rebuttal_report is not None,
        }

    def _aggregate_reviews(self, reviews: List[dict]) -> tuple:
        """Reviewer 리뷰 결과를 규칙 기반으로 집계.

        decision별 가중치:
          agree=+1, partial=0, disagree=-1
        confidence(1~5)를 가중치로 사용.

        Returns:
            (review_score: -1.0~1.0, avg_confidence: 1.0~5.0)
        """
        decision_weight = {"agree": 1.0, "partial": 0.0, "disagree": -1.0}
        total_w, weighted_sum, conf_sum = 0.0, 0.0, 0.0

        for r in reviews:
            conf = r.get("confidence", 3)
            decision = r.get("decision", "partial")
            d_score = decision_weight.get(decision, 0.0)
            weighted_sum += d_score * conf
            total_w += conf
            conf_sum += conf

        if total_w == 0:
            return 0.0, 3.0

        return weighted_sum / total_w, conf_sum / len(reviews)

    def _review_score_to_stance(self, review_score: float, author_report: dict) -> str:
        """리뷰 집계 점수를 stance로 변환.

        - 리뷰어들이 대체로 동의 (score > 0.3) → Author의 stance 유지
        - 리뷰어들이 대체로 반대 (score < -0.3) → Author의 stance 반전
        - 중간 → 중립 stance
        """
        if review_score > 0.3:
            # 리뷰어 동의 → Author stance 유지
            author_stance = author_report.get("stance", "NEUTRAL")
            if author_stance in self.stance_values:
                return author_stance
            return self.stance_values[0]
        elif review_score < -0.3:
            # 리뷰어 반대 → 가장 반대쪽 stance
            return self.stance_values[-1]
        else:
            # 중간 → 중립
            mid = len(self.stance_values) // 2
            return self.stance_values[mid]

    def _format_mars_debate(
        self,
        author_report: dict,
        reviews: List[dict],
        rebuttal_report: Optional[dict] = None,
    ) -> str:
        """MARS 토론 내용을 텍스트로 정리."""
        lines = [
            f"[Author 분석 — {author_report.get('avatar', '')} "
            f"{author_report['agent_name']} ({author_report['role']})]",
            f"  판단: {author_report['stance']} "
            f"(확신도 {author_report['confidence_score']}%)",
            f"  분석: {author_report['analysis'][:500]}",
            f"  핵심: {', '.join(author_report.get('key_points', [])[:5])}",
        ]

        lines.append("\n[전문가 독립 리뷰]")
        for r in reviews:
            lines.append(
                f"\n{r.get('avatar', '')} {r['reviewer_name']} ({r['role']})"
                f"\n  판정: {r['decision']} (신뢰도 {r['confidence']}/5)"
                f"\n  근거: {r['justification'][:300]}"
            )
            if r.get("suggested_revision"):
                lines.append(f"  수정 제안: {r['suggested_revision'][:200]}")
            if r.get("key_concerns"):
                lines.append(f"  핵심 우려: {', '.join(r['key_concerns'][:3])}")

        if rebuttal_report:
            lines.append(
                f"\n[Author Rebuttal — 수정된 분석]"
                f"\n  판단: {rebuttal_report['stance']} "
                f"(확신도 {rebuttal_report['confidence_score']}%)"
                f"\n  수정 분석: {rebuttal_report['analysis'][:400]}"
                f"\n  핵심: {', '.join(rebuttal_report.get('key_points', [])[:3])}"
            )

        return "\n".join(lines)
