"""
research_collector.py — 웹 리서치 데이터 수집 모듈

Tavily Search API를 사용하여 주제 관련 정보를 수집한다.
Tavily는 AI 에이전트 전용 검색 API로, 정제된 텍스트를 반환한다.

폴백 전략:
  1순위: Tavily Search API (무료 1,000회/월)
  2순위: 최소 데이터로 LLM 분석 진행 (검색 없이)
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _search_tavily(
    query: str,
    max_results: int = 8,
    search_depth: str = "advanced",
    api_key: Optional[str] = None,
) -> Dict:
    """Tavily Search API 호출."""
    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python 미설치. pip install tavily-python")
        return {}

    key = api_key or os.environ.get("TAVILY_API_KEY", "")
    if not key:
        logger.warning("TAVILY_API_KEY 미설정")
        return {}

    try:
        client = TavilyClient(api_key=key)
        results = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )
        return results
    except Exception as e:
        logger.error(f"Tavily 검색 실패: {e}")
        return {}


def _format_sources(raw_results: Dict) -> List[Dict]:
    """Tavily 결과를 표준 소스 포맷으로 변환."""
    sources = []
    for r in raw_results.get("results", []):
        sources.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],
            "score": r.get("score", 0),
        })
    return sources


def collect_research(
    topic: str,
    num_results: int = 8,
    tavily_api_key: Optional[str] = None,
) -> Dict:
    """
    주제 기반 웹 리서치 수집.

    두 번의 검색을 수행:
    1. 메인 검색: 주제에 대한 핵심 정보
    2. 반론 검색: 반대 의견, 리스크, 비판

    Args:
        topic: 리서치 주제
        num_results: 메인 검색 결과 수
        tavily_api_key: Tavily API 키 (없으면 환경변수)

    Returns:
        표준 리서치 데이터 dict
    """
    logger.info(f"리서치 수집 시작: '{topic}'")

    # 1차 검색: 메인 토픽
    main_results = _search_tavily(
        query=topic,
        max_results=num_results,
        search_depth="advanced",
        api_key=tavily_api_key,
    )

    # 2차 검색: 반대 의견/리스크
    counter_query = f"{topic} risks challenges criticism limitations 한계 문제점"
    counter_results = _search_tavily(
        query=counter_query,
        max_results=5,
        search_depth="basic",
        api_key=tavily_api_key,
    )

    # 한국어 보강 검색 (주제가 영어일 경우)
    kr_results = {}
    if not any(ord(c) > 0x3000 for c in topic):
        kr_results = _search_tavily(
            query=f"{topic} 한국 동향 전망 분석",
            max_results=4,
            search_depth="basic",
            api_key=tavily_api_key,
        )

    # 데이터 통합
    main_sources = _format_sources(main_results)
    counter_sources = _format_sources(counter_results)
    kr_sources = _format_sources(kr_results)

    # 한국어 소스를 메인에 병합
    if kr_sources:
        main_sources.extend(kr_sources)

    total = len(main_sources) + len(counter_sources)
    logger.info(f"리서치 수집 완료: 총 {total}건 소스")

    return {
        "topic": topic,
        "ai_summary": main_results.get("answer", ""),
        "sources": main_sources,
        "counter_sources": counter_sources,
        "total_sources": total,
        "collected_at": datetime.now().isoformat(),
    }
