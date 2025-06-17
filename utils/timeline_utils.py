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


def available_url(url: str) -> bool:
    publishers = ["chosun", "sbs", "msn.com"]
    for publisher in publishers:
        if publisher in url:
            return False
    return True


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


def shrink_if_needed(strings, threshold=4000):
    total_len = sum(len(s) for s in strings)
    if total_len <= threshold:
        return strings
    # Short enough

    return shrink_if_needed([s for i, s in enumerate(strings) if i % 2 == 0])
