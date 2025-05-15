import requests
from datetime import datetime, timedelta
from utils.timeline_utils import available_url

# ---------------------------------------------------


# 검색어, 시작 날짜, 종료 날짜, API_KEY -> (링크, 제목) 리스트
def get_news_serper(
    query: str,
    date: datetime,
    api_key: str
) -> tuple[str, str]:
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
        for news in result:
            link = news.get("link")
            title = news.get("title")
            if available_url(link):
                break
            else:
                link = ""

        if not link or not title:
            return None
        return (link, title)

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

    # 3개월 이상 차단, 안전장치
    if current < endAt - timedelta(days=90):
        current = endAt-timedelta(days=90)

    # 기사 수집
    while current <= endAt:
        url_title = get_news_serper(query, current, api_key)
        if url_title:
            date_str = current.strftime("%Y-%m-%d")
            results.append((url_title[0], url_title[1], date_str))
        current += timedelta(days=1)

    return results
