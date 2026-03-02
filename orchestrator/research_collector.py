"""
research_collector.py — 웹 리서치 데이터 수집 모듈

Tavily Search API를 사용하여 주제 관련 정보를 수집한다.
Tavily는 AI 에이전트 전용 검색 API로, 정제된 텍스트를 반환한다.

v2 개선사항:
  - include_raw_content=True로 본문 품질 대폭 향상
  - 소스 품질 필터링 (최소 길이, YouTube 제외, 중복 제거)
  - 도메인별 보강 검색 (stock → 시세/환율/금리 정량 데이터)
  - content/raw_content 하이브리드 추출

폴백 전략:
  1순위: Tavily Search API (무료 1,000회/월)
  2순위: 최소 데이터로 LLM 분석 진행 (검색 없이)
"""
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# --- 소스 품질 필터링 설정 ---
MIN_CONTENT_LENGTH = 80  # 최소 콘텐츠 길이 (이하면 필터링)
MAX_CONTENT_LENGTH = 1500  # raw_content에서 추출 시 최대 길이
LOW_QUALITY_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
]
# 커뮤니티/게시판 도메인 (날짜 확인이 더 중요)
COMMUNITY_DOMAINS = [
    "pgr21.com",
    "dcinside.com",
    "fmkorea.com",
    "ruliweb.com",
    "reddit.com",
    "quora.com",
    "namu.wiki",
]

# --- 도메인별 보강 검색 쿼리 ---
DOMAIN_SUPPLEMENTARY_QUERIES = {
    "stock": [
        "코스피 지수 오늘 현재 시세 {year}년 {month}월",
        "원달러 환율 오늘 현재 {year}년 {month}월",
        "한국은행 기준금리 현재 {year}년",
    ],
    "economy": [
        "한국 GDP 성장률 최신 {year}년",
        "소비자물가 CPI 최신 {year}년 {month}월",
        "기준금리 현재 {year}년",
    ],
}


def _search_tavily(
    query: str,
    max_results: int = 8,
    search_depth: str = "advanced",
    api_key: Optional[str] = None,
    include_raw: bool = True,
) -> Dict:
    """Tavily Search API 호출.

    Args:
        include_raw: True면 raw_content 포함 (본문 품질 향상, API 크레딧 소모 증가)
    """
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
            include_raw_content=include_raw,
        )
        return results
    except Exception as e:
        logger.error(f"Tavily 검색 실패: {e}")
        return {}


def _extract_best_content(result: Dict, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Tavily 결과에서 최상의 콘텐츠를 추출.

    raw_content가 있으면 그것을 우선 사용하고,
    없거나 너무 짧으면 content를 사용한다.
    사이드바/네비게이션 쓰레기 텍스트를 필터링한다.
    """
    raw = result.get("raw_content", "") or ""
    content = result.get("content", "") or ""

    # raw_content가 충분히 길면 우선 사용
    if len(raw) > len(content) and len(raw) > MIN_CONTENT_LENGTH:
        text = raw
    else:
        text = content

    # 쓰레기 텍스트 정리: 네비게이션/사이드바 패턴 제거
    text = _clean_content(text)

    # 길이 제한
    if len(text) > max_length:
        # 문장 단위로 자르기
        sentences = re.split(r'(?<=[.!?。])\s+', text[:max_length + 200])
        trimmed = ""
        for s in sentences:
            if len(trimmed) + len(s) > max_length:
                break
            trimmed += s + " "
        text = trimmed.strip() if trimmed.strip() else text[:max_length]

    return text


def _clean_content(text: str) -> str:
    """웹 페이지 크롤링 시 섞여 들어오는 쓰레기 텍스트 제거."""
    # ### 로 시작하는 사이드바/추천 기사 패턴 반복 제거
    # 동일 패턴이 3번 이상 반복되면 사이드바로 판단
    lines = text.split("\n")
    cleaned_lines = []
    sidebar_pattern_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 사이드바 패턴: "###" 로 시작하는 짧은 제목 반복
        if line.startswith("###") and len(line) < 80:
            sidebar_pattern_count += 1
            if sidebar_pattern_count >= 3:
                continue  # 사이드바로 판단, 스킵
        else:
            sidebar_pattern_count = 0
        # 네비게이션 패턴: "|" 가 많은 줄
        if line.count("|") >= 3 and len(line) < 200:
            continue
        # 광고/배너 패턴
        if any(kw in line.lower() for kw in ["광고", "배너", "구독하기", "로그인", "회원가입"]):
            if len(line) < 50:
                continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _is_low_quality_source(source: Dict) -> bool:
    """저품질 소스 판별.

    Returns:
        True면 필터링 대상
    """
    url = source.get("url", "")
    content = source.get("content", "")
    title = source.get("title", "")

    # 1. 최소 콘텐츠 길이 미달
    if len(content) < MIN_CONTENT_LENGTH:
        logger.info(f"소스 필터링 (콘텐츠 짧음): {title[:50]}")
        return True

    # 2. 동영상 플랫폼 (트랜스크립트 추출 불가)
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    if any(d in domain for d in LOW_QUALITY_DOMAINS):
        logger.info(f"소스 필터링 (동영상/SNS): {title[:50]}")
        return True

    # 3. 제목과 본문 불일치 감지 (제목 키워드가 본문에 거의 없는 경우)
    title_keywords = set(re.findall(r'[\w가-힣]+', title.lower()))
    content_lower = content.lower()
    if title_keywords:
        match_ratio = sum(1 for kw in title_keywords if kw in content_lower) / len(title_keywords)
        if match_ratio < 0.15 and len(title_keywords) >= 3:
            logger.info(f"소스 필터링 (제목-본문 불일치 {match_ratio:.0%}): {title[:50]}")
            return True

    return False


def _deduplicate_sources(sources: List[Dict]) -> List[Dict]:
    """중복 URL 제거. 먼저 나온 소스 우선."""
    seen_urls = set()
    unique = []
    for s in sources:
        url = s.get("url", "").split("?")[0].rstrip("/")  # 쿼리 파라미터 무시
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append(s)
    return unique


def _format_sources(raw_results: Dict, apply_filter: bool = True) -> List[Dict]:
    """Tavily 결과를 표준 소스 포맷으로 변환.

    Args:
        apply_filter: True면 저품질 소스 필터링 적용
    """
    sources = []
    for r in raw_results.get("results", []):
        content = _extract_best_content(r)
        source = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": content,
            "score": r.get("score", 0),
        }

        if apply_filter and _is_low_quality_source(source):
            continue

        sources.append(source)

    return sources


def _get_supplementary_queries(domain: str) -> List[str]:
    """도메인별 보강 검색 쿼리 생성."""
    now = datetime.now()
    queries = DOMAIN_SUPPLEMENTARY_QUERIES.get(domain, [])
    return [
        q.format(year=now.year, month=now.month)
        for q in queries
    ]


def collect_research(
    topic: str,
    num_results: int = 8,
    tavily_api_key: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict:
    """
    주제 기반 웹 리서치 수집 (v2).

    검색 전략:
    1. 메인 검색: 주제에 대한 핵심 정보 (raw_content 포함)
    2. 반론 검색: 반대 의견, 리스크, 비판
    3. 도메인 보강 검색: stock → 시세/환율/금리 정량 데이터
    4. 한국어 보강 검색: 영어 주제인 경우

    품질 관리:
    - 저품질 소스 자동 필터링 (YouTube, SNS, 짧은 콘텐츠)
    - 제목-본문 불일치 감지
    - 중복 URL 제거
    - raw_content에서 사이드바/네비게이션 쓰레기 제거

    Args:
        topic: 리서치 주제
        num_results: 메인 검색 결과 수
        tavily_api_key: Tavily API 키 (없으면 환경변수)
        domain: 도메인 힌트 (stock, economy 등) — 보강 검색에 사용

    Returns:
        표준 리서치 데이터 dict
    """
    logger.info(f"리서치 수집 시작: '{topic}' (domain={domain})")

    # 1차 검색: 메인 토픽 (raw_content 포함으로 본문 품질 향상)
    main_results = _search_tavily(
        query=topic,
        max_results=num_results,
        search_depth="advanced",
        api_key=tavily_api_key,
        include_raw=True,
    )

    # 2차 검색: 반대 의견/리스크
    counter_query = f"{topic} risks challenges criticism limitations 한계 문제점 리스크"
    counter_results = _search_tavily(
        query=counter_query,
        max_results=5,
        search_depth="basic",
        api_key=tavily_api_key,
        include_raw=False,  # 반론은 basic으로 충분
    )

    # 3차 검색: 도메인별 보강 (정량 데이터 확보)
    supplementary_sources = []
    if domain:
        supp_queries = _get_supplementary_queries(domain)
        logger.info(f"보강 검색 시작: domain={domain}, queries={len(supp_queries)}개")
        for sq in supp_queries:
            logger.info(f"보강 검색: '{sq}'")
            supp_results = _search_tavily(
                query=sq,
                max_results=3,
                search_depth="basic",
                api_key=tavily_api_key,
                include_raw=True,
            )
            if supp_results:
                formatted = _format_sources(supp_results)
                logger.info(f"  → {len(formatted)}건 수집")
                supplementary_sources.extend(formatted)
            else:
                logger.warning(f"  → 보강 검색 결과 없음: '{sq}'")
    else:
        logger.info("domain 미지정 → 보강 검색 건너뜀")

    # 4차 검색: 한국어 보강 (주제가 영어일 경우)
    kr_results = {}
    if not any(ord(c) > 0x3000 for c in topic):
        kr_results = _search_tavily(
            query=f"{topic} 한국 동향 전망 분석",
            max_results=4,
            search_depth="basic",
            api_key=tavily_api_key,
            include_raw=False,
        )

    # 데이터 통합 + 품질 필터링
    main_sources = _format_sources(main_results)
    counter_sources = _format_sources(counter_results)
    kr_sources = _format_sources(kr_results)

    # 한국어 소스를 메인에 병합
    if kr_sources:
        main_sources.extend(kr_sources)

    # 보강 소스를 별도 카테고리로 유지
    if supplementary_sources:
        supplementary_sources = _deduplicate_sources(supplementary_sources)

    # 전체 중복 제거
    all_urls = {s["url"].split("?")[0].rstrip("/") for s in main_sources}
    counter_sources = [s for s in counter_sources if s["url"].split("?")[0].rstrip("/") not in all_urls]
    supp_urls = all_urls | {s["url"].split("?")[0].rstrip("/") for s in counter_sources}
    supplementary_sources = [s for s in supplementary_sources if s["url"].split("?")[0].rstrip("/") not in supp_urls]

    total = len(main_sources) + len(counter_sources) + len(supplementary_sources)
    logger.info(
        f"리서치 수집 완료: 메인 {len(main_sources)}건, "
        f"반론 {len(counter_sources)}건, "
        f"보강 {len(supplementary_sources)}건 (총 {total}건)"
    )

    result = {
        "topic": topic,
        "ai_summary": main_results.get("answer", ""),
        "sources": main_sources,
        "counter_sources": counter_sources,
        "total_sources": total,
        "collected_at": datetime.now().isoformat(),
    }

    # 보강 소스가 있으면 별도 키로 추가
    if supplementary_sources:
        result["supplementary_sources"] = supplementary_sources

    return result
