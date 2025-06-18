import requests
from datetime import date, timedelta
from utils.timeline_utils import available_url, auto_clean_url

# ---------------------------------------------------

lang_to_country = {
    "en": "us",
    "ko": "kr",
    "ja": "jp",
    "es": "es",
    "fr": "fr",
    "ru": "ru",
}

# ---------------------------------------------------


# 검색어, 시작 날짜, 종료 날짜, API_KEY -> (링크, 제목) 리스트
def get_news_serper(
    query: str,
    date: date,
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
            link = auto_clean_url(news.get("link"))
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
    startAt: date,
    endAt: date,
    api_key: str,
) -> list[tuple[str, str, date]]:
    results = []
    current = startAt

    # 3개월 이상 차단, 안전장치
    if current < endAt - timedelta(days=90):
        current = endAt - timedelta(days=90)

    # 검색어 공백 기준 분할
    query_tokens = query.strip().split()

    # 기사 수집
    current -= timedelta(days=1)
    while current <= endAt:
        # 한 날짜의 여러 뉴스 링크
        max_count = -1
        best_news = None
        seen_links = set()
        current += timedelta(days=1)
        news_list = get_news_serper(query, current, api_key)
        if not news_list:
            continue

        # 검색어가 많이 나타난 뉴스 찾기
        for link, title in news_list:
            if link in seen_links:
                continue

            count = sum(title.count(token) for token in query_tokens)
            if count > max_count:
                best_news = (link, title)
                max_count = count

        # 최적의 뉴스 result에 추가
        link, title = best_news
        seen_links.add(link)
        results.append((link, title, current))

    return results
