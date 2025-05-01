import re
import requests

# ---------------------------------------------------


def get_trending_keywords(api_key):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_trends_trending_now",
        "geo": "KR",
        "hl": "ko",
        "api_key": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch trending keywords: {e}")

    data = response.json()

    # trends 리스트 생성
    trends = []
    for item in data.get("trending_searches", []):
        title = item.get("query")
        if title and re.search(r"[가-힣]", title) and " " not in title:
            trends.append(title)

    return trends
