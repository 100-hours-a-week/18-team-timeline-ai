import requests
from datetime import datetime, timedelta

# ---------------------------------------------------


# 검색어, 시작 날짜, 종료 날짜, API_KEY -> 뉴스 링크 리스트
def get_news_serper(
    query: str,
    date: datetime,
    api_key: str
) -> str:
    # 변수 선언
    date_str = date.strftime("%Y-%m-%d")
    query_with_date = f"{query} {date_str}"

    url = "https://google.serper.dev/news"
    params = {
        "q": query_with_date,
        "hl": "ko",
        "gl": "KR",
        "num": 1,
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
        link = result[0].get("link")
        if not link:
            return []
        return link

    except Exception as e:
        print(f"Serper API 호출 실패: {e}")
        return []


def distribute_news_serper(
    query: str,
    startAt: datetime,
    endAt: datetime,
    api_key: str,
) -> list[tuple[str, datetime]]:
    results = []
    current = startAt

    while current <= endAt:
        partial = get_news_serper(query, current, api_key)
        date_str = current.strftime("%Y-%m-%d")
        results.append((partial, date_str))
        current += timedelta(days=1)

    return results
