import requests
from datetime import datetime
from parse_date import parse_relative_date

# ---------------------------------------------------


# 검색어, 시작 날짜, 종료 날짜, API_KEY -> 뉴스 링크 리스트
def get_news_serper(query: str, startAt: datetime, endAt: datetime, api_key: str) -> list:
    # 변수 선언
    cd_max = endAt.strftime('%m/%d/%Y')
    cd_min = startAt.strftime('%m/%d/%Y')
    num_days = (endAt - startAt).days + 1
    tbs_str = f"cdr:1,cd_min:{cd_min},cd_max:{cd_max}"

    url = "https://google.serper.dev/news"
    params = {
        "q": query,
        "tbs": tbs_str,
        "hl": "ko",
        "gl": "KR",
        "num": num_days * 2,  # 넉넉하게 가져오기
        "api_key": api_key
    }
    headers = {}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        result = response.json().get('news', [])

        # 날짜별 하나씩만 고르기
        date_to_link = {}

        for item in result:
            link = item.get('link')
            date_str = item.get('date')

            if not link or not date_str:
                continue

            try:
                parsed_date = parse_relative_date(date_str)
            except Exception:
                continue

            # 날짜 범위 안에 들어가는 경우만
            if startAt.date() <= parsed_date.date() <= endAt.date():
                day_key = parsed_date.strftime('%Y-%m-%d')
                if day_key not in date_to_link:
                    date_to_link[day_key] = link  # 날짜당 하나만 저장

        # 날짜 순으로 정렬해서 리스트 반환
        sorted_links = [date_to_link[day] for day in sorted(date_to_link.keys())]
        return sorted_links

    except Exception as e:
        print(f"Serper API 호출 실패: {e}")
        return []
