import os
import dotenv
import random
import requests

from datetime import date, timedelta
from utils.timeline_utils import available_url, auto_clean_url

import numpy as np
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------

dotenv.load_dotenv(override=True)
api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")

# ---------------------------------------------------


def get_client():
    key = random.choice(api_keys)
    return genai.Client(api_key=key)


def get_embedding(client: any, text: str) -> list[float]:
    result = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))

    [embedding] = result.embeddings
    return embedding.values


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
            title = news.get("title")
            link = auto_clean_url(news.get("link"))

            if (not title) or (not link):
                continue
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
    while current < endAt:
        # 한 날짜의 여러 뉴스 링크
        max_count = 1
        best_news = []
        current += timedelta(days=1)
        news_list = get_news_serper(query, current, api_key)
        if not news_list:
            continue

        # 검색어가 많이 나타난 뉴스 찾기
        for link, title in news_list:
            count = sum(title.count(token) for token in query_tokens)
            if max_count < count:
                max_count = count
                best_news = []
            if count == max_count:
                best_news.append((link, title))
                max_count = count

        # 관련없는 뉴스만 나옴
        if not best_news:
            continue

        for lt in best_news:
            link, title = lt
            results.append((link, title, current))

    return results


def relevant_news_serper(
    query: str,
    startAt: date,
    endAt: date,
    api_key: str,
) -> list[tuple[str, str, date]]:
    all_results = distribute_news_serper(query, startAt, endAt, api_key)

    # 날짜별로 묶기
    date_grouped = {}
    for link, title, dt in all_results:
        date_grouped.setdefault(dt, []).append((link, title))

    final_results = []
    for dt, items in date_grouped.items():
        if len(items) == 1:
            # 뉴스가 하나면 바로 채택
            final_results.append((items[0][0], items[0][1], dt))
            continue

        # 제목 리스트
        titles = [title for _, title in items]

        # 임베딩 벡터 구하기
        try:
            client = get_client()
            embeddings = [get_embedding(client, title) for title in titles]
        except Exception as e:
            print(f"[{dt}] 임베딩 실패: {e}")
            continue

        # 평균 벡터 계산
        center = np.mean(embeddings, axis=0)

        # 코사인 유사도 계산
        sims = cosine_similarity([center], embeddings)[0]
        if max(sims) < 0.6:
            print(f"[{dt}] 대표 뉴스 없음 (유사도 낮음)")
            continue

        # 가장 중심에 가까운 뉴스 하나 선택
        best_idx = int(np.argmax(sims))
        best_link, best_title = items[best_idx]
        final_results.append((best_link, best_title, dt))

    return final_results
