"""
run_research.py — 멀티에이전트 리서치 파이프라인 진입점

ENGINE_MODE 환경변수로 엔진 선택:
  - "mars" (기본값): MARS 패턴 (Author → Reviewer → Meta-Reviewer)
  - "debate": 기존 3-Phase 토론 (독립 분석 → 교차 반론 → Moderator)

실행 순서:
  1. 환경변수에서 주제(topic), 도메인(domain), 엔진 모드 로드
  2. 웹 리서치 데이터 수집 (Tavily)
  3. 도메인 프리셋 기반 에이전트 동적 생성
  4. 선택된 엔진으로 분석 실행
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
from orchestrator import DebateEngine, MARSEngine, Moderator, collect_research

# ─── 환경 설정 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
ENGINE_MODE = os.environ.get("ENGINE_MODE", "mars").strip().lower()  # "mars" | "debate"
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

def send_telegram(verdict: dict, topic: str, domain_name: str, token: str, chat_id: str,
                   engine_mode: str = "debate"):
    try:
        import requests
        stance = verdict.get("final_stance", "NEUTRAL")
        conf = verdict.get("confidence_score", 50)

        # 스탠스별 이모지 매핑
        stance_upper = stance.upper()
        if stance_upper in ("BUY", "BULLISH", "POSITIVE", "STRONG_AGREE", "AGREE"):
            emoji = "🟢"
        elif stance_upper in ("SELL", "BEARISH", "NEGATIVE", "DISAGREE"):
            emoji = "🔴"
        else:
            emoji = "🟡"

        engine_label = "📝 MARS" if engine_mode == "mars" else "⚔️ Debate"

        msg = (
            f"🧠 <b>Multi-Agent Research Report</b>\n\n"
            f"📌 주제: <b>{topic}</b>\n"
            f"🏷️ 도메인: {domain_name} | 엔진: {engine_label}\n\n"
            f"{emoji} 최종 판단: <b>{stance}</b> "
            f"(확신도 {conf}%)\n\n"
            f"📋 <b>요약</b>\n{verdict.get('summary', '')[:400]}\n\n"
        )

        # MARS 모드: 리뷰어 결정 표시
        if engine_mode == "mars":
            reviewer_decisions = verdict.get("reviewer_decisions", {})
            if reviewer_decisions:
                dec_map = {"agree": "✅", "disagree": "❌", "partial": "⚠️"}
                dec_lines = [f"  {dec_map.get(d, '❓')} {name}: {d}" for name, d in reviewer_decisions.items()]
                msg += f"🔍 <b>리뷰어 결정</b>\n" + "\n".join(dec_lines) + "\n"
            if verdict.get("had_rebuttal"):
                msg += f"🔄 Author Rebuttal 수행됨\n"
            msg += "\n"

        msg += (
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
        "engine_mode": report.get("engine_mode", "debate"),
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
    engine_mode = ENGINE_MODE
    logger.info(f"===== 리서치 파이프라인 시작 =====")
    logger.info(f"주제: {topic}")
    logger.info(f"도메인: {domain}")
    logger.info(f"엔진 모드: {engine_mode}")

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
        domain=domain,
    )
    logger.info(f"수집 완료: {research_data.get('total_sources', 0)}건 소스")

    # 4-1) 도메인별 정량 시장 데이터 수집 (stock, economy)
    try:
        from orchestrator.market_data_collector import collect_market_snapshot
        logger.info(f"[Step 1-1] 시장 정량 데이터 수집 중... (domain={domain})")
        market_snapshot = collect_market_snapshot(domain)
        if market_snapshot:
            research_data["market_snapshot"] = market_snapshot
            snapshot_keys = list(market_snapshot.keys())
            stock_count = len(market_snapshot.get("top_stocks", []))
            logger.info(f"시장 정량 데이터 수집 완료: keys={snapshot_keys}, stocks={stock_count}")
        else:
            logger.warning(f"market_snapshot이 None 반환됨 (domain={domain}은 수집 대상 아닐 수 있음)")
    except ImportError as e:
        logger.error(f"market_data_collector 모듈 로드 실패: {e}")
        logger.error("yfinance/numpy 설치 확인 필요: pip install yfinance numpy")
    except Exception as e:
        logger.error(f"시장 데이터 수집 실패 (계속 진행): {type(e).__name__}: {e}")

    # 5) Moderator (두 모드 공통)
    moderator = Moderator(
        client=client,
        model=MODEL,
        stance_score=preset["stance_score"],
        stance_thresholds=preset["stance_thresholds"],
        stance_values=preset["stance_values"],
    )

    # 6) 엔진 모드에 따라 분기
    if engine_mode == "mars" and "mars_config" in preset:
        # ─── MARS 모드: Author → Reviewer → Meta-Reviewer ──────────────
        logger.info("[Step 2] MARS 엔진으로 분석 진행 중...")
        mars_cfg = preset["mars_config"]

        # Author 에이전트 생성
        author = DynamicAgent(
            client=client,
            model=MODEL,
            agent_cfg=mars_cfg["author"],
            stance_values=preset["stance_values"],
            role_type="author",
        )

        # Reviewer 에이전트 생성 (기존 agents를 reviewer로 활용)
        reviewers = [
            DynamicAgent(
                client=client,
                model=MODEL,
                agent_cfg=agent_cfg,
                stance_values=preset["stance_values"],
                role_type="reviewer",
            )
            for agent_cfg in preset["agents"]
        ]

        # MARS 파이프라인 실행
        engine = MARSEngine(
            author=author,
            reviewers=reviewers,
            enable_rebuttal=True,
            rebuttal_threshold=0.5,
        )
        mars_result = engine.run(research_data)

        # Meta-Reviewer 종합 판단
        logger.info("[Step 3] Meta-Reviewer 종합 판단 중...")
        verdict = moderator.meta_review(
            author_report=mars_result["author_report"],
            reviews=mars_result["reviews"],
            research_data=research_data,
            rebuttal_report=mars_result.get("rebuttal_report"),
        )

        # 보고서 조립 (MARS 구조)
        debate_section = {
            "engine_mode": "mars",
            "author_report": mars_result["author_report"],
            "reviews": mars_result["reviews"],
            "rebuttal_report": mars_result.get("rebuttal_report"),
            "token_efficiency": mars_result.get("token_efficiency", {}),
        }

    else:
        # ─── 기존 Debate 모드: Phase 1 + Phase 2 + Phase 3 ────────────
        if engine_mode == "mars":
            logger.warning("mars_config 없음, debate 모드로 폴백")
        logger.info("[Step 2] Debate 엔진으로 토론 진행 중...")

        agents = [
            DynamicAgent(
                client=client,
                model=MODEL,
                agent_cfg=agent_cfg,
                stance_values=preset["stance_values"],
            )
            for agent_cfg in preset["agents"]
        ]

        engine = DebateEngine(agents, preset["critique_pairs"])
        debate_result = engine.run(research_data)

        logger.info("[Step 3] Moderator 종합 판단 중...")
        verdict = moderator.synthesize(
            reports=debate_result["phase1_reports"],
            critiques=debate_result["phase2_critiques"],
            research_data=research_data,
        )

        # 보고서 조립 (기존 구조)
        debate_section = {
            "engine_mode": "debate",
            "phase1_reports": debate_result["phase1_reports"],
            "phase2_critiques": debate_result["phase2_critiques"],
        }

    # 7) 보고서 조립
    report = {
        "id": timestamp,
        "topic": topic,
        "domain": domain,
        "domain_name": preset["name"],
        "engine_mode": debate_section.get("engine_mode", engine_mode),
        "generated_at": datetime.now().isoformat(),
        "research_data": {
            "topic": research_data.get("topic", ""),
            "ai_summary": research_data.get("ai_summary", ""),
            "total_sources": research_data.get("total_sources", 0),
            "sources": research_data.get("sources", [])[:10],
            "counter_sources": research_data.get("counter_sources", [])[:5],
            "supplementary_sources": research_data.get("supplementary_sources", [])[:6],
            "market_snapshot": research_data.get("market_snapshot"),
            "collected_at": research_data.get("collected_at", ""),
        },
        "debate": debate_section,
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
        send_telegram(verdict, topic, preset["name"], cfg["telegram_token"], cfg["telegram_chat_id"],
                      engine_mode=engine_mode)

    logger.info("===== 파이프라인 완료 =====")
    logger.info(f"최종 판단: {verdict.get('final_stance')} (확신도 {verdict.get('confidence_score')}%)")
    return report


if __name__ == "__main__":
    main()
