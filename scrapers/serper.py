import requests
from datetime import datetime, timedelta
from utils.timeline_utils import available_url

# ---------------------------------------------------


# 검색어, 시작 날짜, 종료 날짜, API_KEY -> (링크, 제목) 리스트
def get_news_serper(
    query: str,
    date: datetime,
    api_key: str
) -> list[tuple[str, str]]:
    # 변수 선언
    date_str = date.strftime("%Y-%m-%d")
    query_with_date = f"{query} {date_str}"

    url = "https://google.serper.dev/news"
    params = {
        "q": query_with_date,
        "hl": "ko",
        "gl": "KR",
        "num": 10,
        "api_key": api_key,
    }

    try:
        # Getting Serper response
        response = requests.get(url, headers={}, params=params)
        response.raise_for_status()
        result = response.json().get("news", [])
        if not result:
            return []

        # Getting news URL
        valid_news = []
        for news in result:
            link = news.get("link")
            title = news.get("title")
            if available_url(link):
                valid_news.append((link, title))

        # Result
        return valid_news

    except Exception as e:
        print(f"Serper API 호출 실패: {e}")
        return None


def distribute_news_serper(
    query: str,
    startAt: datetime,
    endAt: datetime,
    api_key: str,
) -> list[tuple[str, str, datetime]]:
    results = []
    current = startAt
    seen_links = set()

    # 3개월 이상 차단, 안전장치
    if current < endAt - timedelta(days=90):
        current = endAt - timedelta(days=90)

    # 기사 수집
    while current <= endAt:
        news_list = get_news_serper(query, current, api_key)
        for link, title in news_list:
            if link not in seen_links:
                seen_links.add(link)
                results.append((link, title, current.strftime("%Y-%m-%d")))
                break  # 날짜당 하나만 고르도록 유지
        current += timedelta(days=1)

    return results
