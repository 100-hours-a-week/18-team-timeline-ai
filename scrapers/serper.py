import requests
from datetime import datetime, timedelta

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
        title = result[0].get("title")
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

    while current <= endAt:
        url_title = get_news_serper(query, current, api_key)
        if not url_title:
            continue
        date_str = current.strftime("%Y-%m-%d")
        results.append((url_title[0], url_title[1], date_str))
        current += timedelta(days=1)

    return results
