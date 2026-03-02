"""
debate_engine.py — Phase 1 (독립 분석) + Phase 2 (교차 반론) 실행

Korean Stock Agent의 검증된 패턴을 기반으로,
CRITIQUE_PAIRS를 도메인 프리셋에서 동적으로 받아 실행한다.
"""
import logging
from typing import List, Tuple

from agents.base_agent import BaseAgent, AgentReport, AgentCritique

logger = logging.getLogger(__name__)


class DebateEngine:
    def __init__(self, agents: List[BaseAgent], critique_pairs: List[Tuple[int, int]]):
        self.agents = agents
        self.critique_pairs = critique_pairs

    def run(self, research_data: dict) -> dict:
        """
        Phase 1 + Phase 2 실행.

        Returns:
            {
                "phase1_reports": [AgentReport.to_dict(), ...],
                "phase2_critiques": [AgentCritique.to_dict(), ...],
            }
        """
        # ─── Phase 1: 독립 분석 ─────────────────────────────────────────────
        logger.info("=== Phase 1: 독립 분석 시작 ===")
        reports: List[AgentReport] = []

        for agent in self.agents:
            try:
                logger.info(f"  {agent.avatar} {agent.name} 분석 중...")
                report = agent.analyze(research_data)
                reports.append(report)
                logger.info(
                    f"  {agent.avatar} {agent.name}: {report.stance} "
                    f"(확신도 {report.confidence_score}%)"
                )
            except Exception as e:
                logger.error(f"  [{agent.name}] 분석 실패: {e}")
                # 실패 시 중립 기본값으로 삽입 (가중치 0)
                reports.append(AgentReport(
                    agent_name=agent.name,
                    role=agent.role,
                    avatar=agent.avatar,
                    analysis=f"분석 중 오류 발생: {str(e)[:100]}",
                    key_points=["분석 불가"],
                    confidence_score=0,
                    stance="NEUTRAL",
                ))

        # ─── Phase 2: 교차 반론 ─────────────────────────────────────────────
        logger.info("=== Phase 2: 교차 반론 시작 ===")
        critiques: List[AgentCritique] = []

        for from_idx, to_idx in self.critique_pairs:
            if from_idx >= len(self.agents) or to_idx >= len(reports):
                continue
            try:
                from_agent = self.agents[from_idx]
                to_report = reports[to_idx]
                logger.info(
                    f"  {from_agent.avatar} {from_agent.name} → "
                    f"{to_report.avatar} {to_report.agent_name} 반론 중..."
                )
                critique = from_agent.critique(to_report, research_data)
                critiques.append(critique)
            except Exception as e:
                logger.error(
                    f"  [{self.agents[from_idx].name}→{reports[to_idx].agent_name}] "
                    f"반론 실패: {e}"
                )
                critiques.append(AgentCritique(
                    from_agent=self.agents[from_idx].name,
                    to_agent=reports[to_idx].agent_name,
                    critique=f"반론 생성 실패: {str(e)[:80]}",
                ))

        logger.info(
            f"토론 완료: Phase1 {len(reports)}건, Phase2 {len(critiques)}건"
        )

        return {
            "phase1_reports": [r.to_dict() for r in reports],
            "phase2_critiques": [c.to_dict() for c in critiques],
        }
