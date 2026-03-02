"""
mars_engine.py — MARS(Multi-Agent Review System) 파이프라인

학술 리뷰 프로세스에서 영감을 받은 구조:
  Step 1: Author Agent — 리서치 데이터 기반 종합 분석 생성
  Step 2: Reviewer Agents (병렬) — Author 분석을 전문 관점에서 독립 리뷰
  Step 3: (선택) Rebuttal — 리뷰 결과에 따라 Author가 분석 수정

핵심 장점:
  - Reviewer 간 상호작용 제거 → 토큰 ~50% 절감
  - 선형 확장성: O(m) vs MAD의 O(m²)
  - 기존 debate_engine.py는 보존 (fallback 용도)

Reference: MARS: toward more efficient multi-agent collaboration
           for LLM reasoning (arXiv 2509.20502)
"""
import logging
from typing import List

from agents.base_agent import BaseAgent, AgentReport, ReviewerReport

logger = logging.getLogger(__name__)


class MARSEngine:
    """MARS 패턴 기반 멀티에이전트 리서치 엔진.

    Args:
        author: Author 역할의 DynamicAgent (종합 분석 담당)
        reviewers: Reviewer 역할의 DynamicAgent 리스트 (독립 리뷰 담당)
        enable_rebuttal: True이면 리뷰 결과에 따라 Author 재분석 실행
        rebuttal_threshold: disagree 비율이 이 값 이상이면 rebuttal 발동 (0.0~1.0)
    """

    def __init__(
        self,
        author: BaseAgent,
        reviewers: List[BaseAgent],
        enable_rebuttal: bool = True,
        rebuttal_threshold: float = 0.5,
    ):
        self.author = author
        self.reviewers = reviewers
        self.enable_rebuttal = enable_rebuttal
        self.rebuttal_threshold = rebuttal_threshold

    def run(self, research_data: dict) -> dict:
        """MARS 3-Step 파이프라인 실행.

        Returns:
            {
                "author_report": AgentReport.to_dict(),
                "reviews": [ReviewerReport.to_dict(), ...],
                "rebuttal_report": AgentReport.to_dict() | None,
                "engine_mode": "mars",
                "token_efficiency": {
                    "total_llm_calls": int,
                    "author_calls": int,
                    "reviewer_calls": int,
                },
            }
        """
        llm_calls = {"author_calls": 0, "reviewer_calls": 0}

        # ─── Step 1: Author 종합 분석 ──────────────────────────────────────
        logger.info("=== MARS Step 1: Author 종합 분석 ===")
        try:
            logger.info(f"  {self.author.avatar} {self.author.name} 분석 중...")
            author_report = self.author.analyze(research_data)
            llm_calls["author_calls"] += 1
            logger.info(
                f"  {self.author.avatar} {self.author.name}: "
                f"{author_report.stance} (확신도 {author_report.confidence_score}%)"
            )
        except Exception as e:
            logger.error(f"  Author 분석 실패: {e}")
            author_report = AgentReport(
                agent_name=self.author.name,
                role=self.author.role,
                avatar=self.author.avatar,
                analysis=f"Author 분석 중 오류 발생: {str(e)[:200]}",
                key_points=["분석 불가"],
                confidence_score=0,
                stance="NEUTRAL",
            )
            llm_calls["author_calls"] += 1

        # ─── Step 2: 독립 병렬 리뷰 ────────────────────────────────────────
        logger.info(f"=== MARS Step 2: 독립 리뷰 ({len(self.reviewers)}명) ===")
        reviews: List[ReviewerReport] = []

        for reviewer in self.reviewers:
            try:
                logger.info(f"  {reviewer.avatar} {reviewer.name} 리뷰 중...")
                review = reviewer.review(author_report, research_data)
                reviews.append(review)
                llm_calls["reviewer_calls"] += 1
                logger.info(
                    f"  {reviewer.avatar} {reviewer.name}: "
                    f"{review.decision} (신뢰도 {review.confidence}/5)"
                )
            except Exception as e:
                logger.error(f"  [{reviewer.name}] 리뷰 실패: {e}")
                reviews.append(ReviewerReport(
                    reviewer_name=reviewer.name,
                    role=reviewer.role,
                    avatar=reviewer.avatar,
                    decision="partial",
                    confidence=1,
                    justification=f"리뷰 중 오류 발생: {str(e)[:150]}",
                    suggested_revision="",
                    key_concerns=["리뷰 불가"],
                ))
                llm_calls["reviewer_calls"] += 1

        # ─── Step 3: Rebuttal (선택) ───────────────────────────────────────
        rebuttal_report = None

        if self.enable_rebuttal and reviews:
            disagree_count = sum(1 for r in reviews if r.decision == "disagree")
            disagree_ratio = disagree_count / len(reviews)
            logger.info(
                f"=== MARS Step 3: Rebuttal 판단 "
                f"(반대 {disagree_count}/{len(reviews)} = {disagree_ratio:.1%}) ==="
            )

            if disagree_ratio >= self.rebuttal_threshold:
                logger.info("  → Rebuttal 발동: Author 재분석 실행")
                rebuttal_report = self._run_rebuttal(
                    author_report, reviews, research_data
                )
                llm_calls["author_calls"] += 1
            else:
                logger.info("  → Rebuttal 불필요: 충분한 합의")

        total_calls = llm_calls["author_calls"] + llm_calls["reviewer_calls"]
        logger.info(
            f"MARS 완료: Author {llm_calls['author_calls']}회, "
            f"Reviewer {llm_calls['reviewer_calls']}회, "
            f"총 {total_calls}회 LLM 호출"
        )

        return {
            "author_report": author_report.to_dict(),
            "reviews": [r.to_dict() for r in reviews],
            "rebuttal_report": rebuttal_report.to_dict() if rebuttal_report else None,
            "engine_mode": "mars",
            "token_efficiency": {
                "total_llm_calls": total_calls,
                **llm_calls,
            },
        }

    def _run_rebuttal(
        self,
        original_report: AgentReport,
        reviews: List[ReviewerReport],
        research_data: dict,
    ) -> AgentReport:
        """리뷰 피드백을 반영하여 Author가 분석을 수정한다.

        MARS 논문의 Rebuttal 안전장치:
        "If you strongly agree with the meta-reviewer's suggestions,
         revise your answer accordingly. If you disagree, insist on
         your initial answer and repeat it."
        """
        # 리뷰 피드백 텍스트 구성
        feedback_lines = []
        for r in reviews:
            feedback_lines.append(
                f"[{r.reviewer_name} ({r.role})] "
                f"판정: {r.decision} (신뢰도 {r.confidence}/5)\n"
                f"근거: {r.justification[:300]}\n"
                f"수정 제안: {r.suggested_revision[:200]}\n"
                f"핵심 우려: {', '.join(r.key_concerns[:3])}"
            )
        feedback_text = "\n\n".join(feedback_lines)

        # Rebuttal 프롬프트 (2-turn conversation)
        messages = [
            {
                "role": "user",
                "content": f"""아래는 당신의 초기 분석에 대한 전문가 리뷰 결과입니다.

[당신의 초기 분석]
판단: {original_report.stance} (확신도: {original_report.confidence_score}%)
분석: {original_report.analysis}
핵심: {', '.join(original_report.key_points[:5])}

[전문가 리뷰 피드백]
{feedback_text}

[Rebuttal 가이드]
- 리뷰어들의 지적이 타당하다면, 해당 부분을 반영하여 분석을 수정하세요.
- 리뷰어의 지적에 동의하지 않는다면, 반박 근거를 명확히 하고 기존 분석을 유지하세요.
- 수정 시 원래 분석의 강점은 보존하면서 약점만 개선하세요.

반드시 아래 JSON으로만 응답:
{{
  "analysis": "수정된 종합 분석 (500자 이상, 리뷰 반영 사항 포함)",
  "key_points": ["수정된 인사이트 1", "수정된 인사이트 2", "수정된 인사이트 3"],
  "confidence_score": 75,
  "stance": "{original_report.stance}"
}}

stance: {' / '.join(self.author.stance_values)} 중 하나""",
            }
        ]

        try:
            result = self.author._call_llm(messages, max_tokens=3072)
            data = self.author._parse_json_response(result)

            raw_stance = data.get("stance", original_report.stance).upper()
            if raw_stance not in self.author.stance_values:
                raw_stance = original_report.stance

            return AgentReport(
                agent_name=f"{self.author.name} (수정)",
                role=self.author.role,
                avatar=self.author.avatar,
                analysis=data.get("analysis", result[:600]),
                key_points=data.get("key_points", original_report.key_points),
                confidence_score=max(0, min(100, int(data.get("confidence_score", 50)))),
                stance=raw_stance,
            )
        except Exception as e:
            logger.error(f"Rebuttal 실패, 원본 유지: {e}")
            return original_report
