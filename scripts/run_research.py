"""
run_research.py — 멀티에이전트 리서치 파이프라인 진입점

실행 순서:
  1. 환경변수에서 주제(topic)와 도메인(domain) 로드
  2. 웹 리서치 데이터 수집 (Tavily)
  3. 도메인 프리셋 기반 에이전트 동적 생성
  4. 3-Phase 토론 실행
  5. JSON 보고서 생성 + 히스토리 아카이브
  6. Telegram 알림 발송
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 파이썬 경로에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import anthropic

from agents import DynamicAgent
from config.domains import get_domain_preset
from orchestrator import DebateEngine, Moderator, collect_research

# ─── 환경 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
DOCS_DIR = ROOT / "docs" / "data"
HISTORY_DIR = ROOT / "data" / "history"


# ─── 설정 로드 ────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg = {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "tavily_api_key":    os.environ.get("TAVILY_API_KEY", ""),
        "telegram_token":    os.environ.get("TELEGRAM_TOKEN", ""),
        "telegram_chat_id":  os.environ.get("TELEGRAM_CHAT_ID", ""),
    }
    config_path = ROOT / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if not cfg.get(k):
                    cfg[k] = v
    if not cfg["anthropic_api_key"]:
        raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
    return cfg


# ─── Telegram 알림 ────────────────────────────────────────────────────────────

def send_telegram(verdict: dict, topic: str, domain_name: str, token: str, chat_id: str):
    try:
        import requests
        stance = verdict.get("final_stance", "NEUTRAL")
        conf = verdict.get("confidence_score", 50)

        # 도메인별 이모지 매핑
        stance_upper = stance.upper()
        if stance_upper in ("BUY", "BULLISH", "POSITIVE", "STRONG_AGREE", "AGREE"):
            emoji = "🟢"
        elif stance_upper in ("SELL", "BEARISH", "NEGATIVE", "DISAGREE"):
            emoji = "🔴"
        else:
            emoji = "🟡"

        msg = (
            f"🧠 <b>Multi-Agent Research Report</b>\n\n"
            f"📌 주제: <b>{topic}</b>\n"
            f"🏷️ 도메인: {domain_name}\n\n"
            f"{emoji} 최종 판단: <b>{stance}</b> "
            f"(확신도 {conf}%)\n\n"
            f"📋 <b>요약</b>\n{verdict.get('summary', '')[:400]}\n\n"
            f"💡 인사이트: {', '.join(verdict.get('key_insights', [])[:3])}\n"
            f"⚠️ 리스크: {', '.join(verdict.get('risk_factors', [])[:3])}\n\n"
            f"📎 대시보드: https://jinhae8971.github.io/multi-agent-researcher/"
        )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=20)
        logger.info("Telegram 알림 발송 완료")
    except Exception as e:
        logger.warning(f"Telegram 발송 실패 (무시): {e}")


# ─── 히스토리 관리 ────────────────────────────────────────────────────────────

def update_history_index(report: dict, timestamp: str):
    """히스토리 인덱스 파일 업데이트. 대시보드에서 과거 보고서 목록을 표시할 때 사용."""
    index_path = DOCS_DIR / "history" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    index = []
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = []

    verdict = report.get("verdict", {})
    entry = {
        "id": timestamp,
        "topic": report.get("topic", ""),
        "domain": report.get("domain", ""),
        "domain_name": report.get("domain_name", ""),
        "final_stance": verdict.get("final_stance", ""),
        "confidence_score": verdict.get("confidence_score", 0),
        "summary": verdict.get("summary", "")[:150],
        "generated_at": report.get("generated_at", ""),
    }

    # 최신순 삽입, 최대 50개 유지
    index.insert(0, entry)
    index = index[:50]

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


# ─── 메인 파이프라인 ──────────────────────────────────────────────────────────

def main():
    # 1) 입력 파라미터
    topic = os.environ.get("RESEARCH_TOPIC", "").strip()
    domain = os.environ.get("RESEARCH_DOMAIN", "general").strip().lower()

    if not topic:
        logger.error("RESEARCH_TOPIC 환경변수가 비어있습니다.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"===== 리서치 파이프라인 시작 =====")
    logger.info(f"주제: {topic}")
    logger.info(f"도메인: {domain}")

    # 2) 설정 로드
    cfg = load_config()
    client = anthropic.Anthropic(api_key=cfg["anthropic_api_key"])

    # 3) 도메인 프리셋 로드
    preset = get_domain_preset(domain)
    logger.info(f"도메인 프리셋: {preset['name']} (에이전트 {len(preset['agents'])}명)")

    # 4) 웹 리서치 수집
    logger.info("[Step 1] 웹 리서치 수집 중...")
    research_data = collect_research(
        topic=topic,
        num_results=8,
        tavily_api_key=cfg.get("tavily_api_key"),
    )
    logger.info(f"수집 완료: {research_data.get('total_sources', 0)}건 소스")

    # 5) 에이전트 동적 생성
    agents = [
        DynamicAgent(
            client=client,
            model=MODEL,
            agent_cfg=agent_cfg,
            stance_values=preset["stance_values"],
        )
        for agent_cfg in preset["agents"]
    ]

    # 6) Phase 1 + Phase 2 토론
    logger.info("[Step 2] 에이전트 토론 진행 중...")
    engine = DebateEngine(agents, preset["critique_pairs"])
    debate_result = engine.run(research_data)

    # 7) Phase 3: Moderator 종합 판단
    logger.info("[Step 3] Moderator 종합 판단 중...")
    moderator = Moderator(
        client=client,
        model=MODEL,
        stance_score=preset["stance_score"],
        stance_thresholds=preset["stance_thresholds"],
        stance_values=preset["stance_values"],
    )
    verdict = moderator.synthesize(
        reports=debate_result["phase1_reports"],
        critiques=debate_result["phase2_critiques"],
        research_data=research_data,
    )

    # 8) 보고서 조립
    report = {
        "id": timestamp,
        "topic": topic,
        "domain": domain,
        "domain_name": preset["name"],
        "generated_at": datetime.now().isoformat(),
        "research_data": {
            "topic": research_data.get("topic", ""),
            "ai_summary": research_data.get("ai_summary", ""),
            "total_sources": research_data.get("total_sources", 0),
            "sources": research_data.get("sources", [])[:10],
            "counter_sources": research_data.get("counter_sources", [])[:5],
            "collected_at": research_data.get("collected_at", ""),
        },
        "debate": {
            "phase1_reports": debate_result["phase1_reports"],
            "phase2_critiques": debate_result["phase2_critiques"],
        },
        "verdict": verdict,
    }

    # 9) 저장
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # latest_report.json (대시보드용)
    latest_path = DOCS_DIR / "latest_report.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"최신 보고서 저장: {latest_path}")

    # 히스토리 아카이브
    archive_path = HISTORY_DIR / f"{timestamp}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 히스토리에서 docs/data/history/로도 복사 (Pages에서 접근 가능)
    docs_archive = DOCS_DIR / "history" / f"{timestamp}.json"
    docs_archive.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_archive, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 히스토리 인덱스 업데이트
    update_history_index(report, timestamp)

    # 10) Telegram 알림
    if cfg.get("telegram_token") and cfg.get("telegram_chat_id"):
        send_telegram(verdict, topic, preset["name"], cfg["telegram_token"], cfg["telegram_chat_id"])

    logger.info("===== 파이프라인 완료 =====")
    logger.info(f"최종 판단: {verdict.get('final_stance')} (확신도 {verdict.get('confidence_score')}%)")
    return report


if __name__ == "__main__":
    main()
