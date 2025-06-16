import re
from urllib.parse import urlparse, parse_qs, unquote
from util.logger import Logger
import logging

logger = Logger.get_logger("url_utils", log_level=logging.INFO)


def clean_url(url: str) -> str:
    """
    URL을 정리하는 함수

    Args:
        url: 정리할 URL

    Returns:
        str: 정리된 URL
    """
    try:
        # URL 디코딩
        url = unquote(url)

        # HTML 태그 제거 (더 강력한 패턴)
        url = re.sub(r"<[^>]*>", "", url)
        url = re.sub(r"&[a-zA-Z0-9#]+;", "", url)  # HTML 엔티티 제거

        # URL 파싱
        parsed = urlparse(url)

        # 쿼리 파라미터 정리
        if parsed.query:
            params = parse_qs(parsed.query)
            # 필요한 파라미터만 유지 (ID 또는 기본 파라미터)
            clean_params = {}
            for key, value in params.items():
                if key.lower() in ["id", "articleid", "newsid", "seq"]:
                    # ID 값에서 HTML 태그와 특수문자 제거
                    clean_value = re.sub(r"[^a-zA-Z0-9]", "", value[0])
                    if clean_value:
                        clean_params[key] = clean_value

            if clean_params:
                # 쿼리 문자열 재구성
                query = "&".join(f"{k}={v}" for k, v in clean_params.items())
                # URL 재구성
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{query}"
            else:
                # 유효한 파라미터가 없으면 쿼리스트링 제거
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # 최종 URL 정리
        url = re.sub(r"\s+", "", url)  # 공백 제거
        url = re.sub(r"[^\w\-\.\:\/\?\=\&]", "", url)  # URL에 허용된 문자만 유지

        return url

    except Exception as e:
        logger.error(f"[URL Utils] URL 정리 실패: {url}, 에러: {str(e)}")
        return url
