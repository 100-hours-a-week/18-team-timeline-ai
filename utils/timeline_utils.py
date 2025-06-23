import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

noise_prefixes = [
    "utm_",
    "ref",
    "search_",
    "session",
    "tracking_",
    "gclid",
    "fbclid",
    "adid",
    "clid",
]


def next_timeline_type(current: str) -> str:
    order = ["DAY", "WEEK", "MONTH"]
    if current not in order:
        return "ERROR"

    index = order.index(current)
    return order[min(index + 1, len(order) - 1)]


def convert_tag(tag: str) -> int:
    if "경제" in tag:
        return 1
    elif "연예" in tag:
        return 2
    elif "스포츠" in tag:
        return 3
    else:
        return 0


# 스크래핑이 차단되거나, 뉴스사가 아닌 링크들을 배제
def available_url(url: str) -> bool:
    publishers = ["chosun", "sbs", "msn.com", "worktoday", "kspnews",
                  "thisisgame", "artinsight", "footboom", "koreatimes",
                  "breaknews", "ecojournal", "anewsa.com", "koreadaily",
                  "100news", "blog", "tistory", "theqoo", "dcinside",
                  "jajusibo", "vop.co.kr", "jeongpil"]
    for publisher in publishers:
        if publisher in url:
            return False
    return True


# 인명을 검색했는데, 기자 이름을 인식한 경우를 탐지
def reporter_issue(query: str, snippet: str) -> bool:
    reporter_names = re.findall(r'([가-힣]{2,4})\s*기자', snippet)
    for name in reporter_names:
        if name in query:
            return True
    return False


def remove_sidebar_noise(text: str) -> str:
    # '최신뉴스', '베스트클릭' 등과 같이 본문 외 사이드 뉴스 이후의 텍스트를 제거합니다.

    noise_keywords = ["최신뉴스", "베스트클릭", "관련기사", "많이 본 뉴스", "함께 본 뉴스", "주요뉴스", "핫클릭"]

    lines = text.splitlines()
    clean_lines = []

    for line in lines:
        if any(keyword in line for keyword in noise_keywords):
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


# 외국어 기사 필터링용
def contains_korean(title: str) -> bool:
    return bool(re.search(r'[가-힣]', title))


# 기사 링크에 붙은 불필요한 파라미터를 제거
def auto_clean_url(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)

    if not query:
        return url

    filtered_query = {
        k: v
        for k, v in query.items()
        if not any(k.lower().startswith(prefix) for prefix in noise_prefixes)
    }

    new_query = urlencode(filtered_query, doseq=True)
    cleaned_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
    ))

    return cleaned_url


# 제목을 짧게 자르기 위한 함수
def short_sentence(text: str) -> str:
    # 각종 괄호 제거
    patterns = [r"\[.*?\]", r"\(.*?\)", r"<.*?>", r"【.*?】", r"《.*?》"]
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    # 마침표 뒤 제거 (소수점은 보호)
    text = re.split(r"(?<=[^0-9])\.\s+", text)[0].strip()

    # 생략부호 뒤 제거
    if "…" in text:
        text = text.split("…")[0].strip()

    # 전각 뒤 제거
    if "|" in text:
        text = text.split("|")[0].strip()
    if "｜" in text:
        text = text.split("｜")[0].strip()

    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text.strip()


# 요약본을 특정 길이로 자연스럽게 자르는 함수
def compress_sentence(text: str, target_len: int = 70) -> str:
    # 마침표 기준으로 문장 분할
    sentences = re.split(r"[.…]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # 특정 길이 이상이면 스캔 멈춤
    result = ""
    for sentence in sentences:
        result += sentence + ". "
        if len(result) >= target_len:
            break

    return result.strip()


# AI 입력을 위해 기사 원본을, 문맥 무시하고 자르는 함수
def shrink_if_needed(strings, threshold=4000):
    total_len = sum(len(s) for s in strings)
    if total_len <= threshold:
        return strings
    # Short enough

    return shrink_if_needed([s for i, s in enumerate(strings) if i % 2 == 0])
