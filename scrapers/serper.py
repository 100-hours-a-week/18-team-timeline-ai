import os
import requests
import json
from datetime import datetime, timedelta

# ---------------------------------------------------

def get_news_serper(query: str, cnt: int, SERPER_API_KEY: str) -> list:
    today = datetime.today()
    cd_max = today.strftime("%Y-%m-%d")
    cd_min = (today - timedelta(days=cnt)).strftime("%Y-%m-%d")

    tbs_str = f"cdr:1,cd_min:{cd_min},cd_max:{cd_max}"

    url = "https://google.serper.dev/news"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "tbs": tbs_str, "hl": "ko", "gl": "KR", "num": cnt}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 요청 실패시 예외 발생
        data = response.json()

        news_items = data.get("news", [])
        links = [item["link"] for item in news_items if "link" in item]
        return links

    except Exception as e:
        print(f"Serper API 호출 실패: {e}")
        return []
