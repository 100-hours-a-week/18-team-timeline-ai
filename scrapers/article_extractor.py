import aiohttp
import logging
import asyncio
import numpy as np
import trafilatura
from trafilatura.settings import use_config
from util.logger import Logger
from config.settings import (
    USER_AGENT,
    ARTICLE_TIMEOUT,
)
from typing import List, Dict, Optional, AsyncGenerator
from scrapers.base_searcher import BaseSearcher
from urllib.parse import urlparse

logger = Logger.get_logger("article_extractor", log_level=logging.INFO)
config = use_config()
config.set("DEFAULT", "user-agent", USER_AGENT)
headers = {"User-Agent": USER_AGENT}


class ArticleExtractor(BaseSearcher):
    def __init__(self, lang: str = "ko", max_workers: int = 6):
        self.max_workers = max_workers
        self.lang = lang
        self.session = None
        self.domain_timeouts = {
            "sportivomedia.net": 30,
            "default": ARTICLE_TIMEOUT,
        }
        self.domain_retries = {
            "sportivomedia.net": 5,
            "default": 3,
        }

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            connector=aiohttp.TCPConnector(
                limit=self.max_workers, force_close=True, enable_cleanup_closed=True
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.session.closed:
            await self.session.close()

    def _get_timeout_for_domain(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            return self.domain_timeouts.get(domain, self.domain_timeouts["default"])
        except Exception:
            return self.domain_timeouts["default"]

    def _get_retries_for_domain(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            return self.domain_retries.get(domain, self.domain_retries["default"])
        except Exception:
            return self.domain_retries["default"]

    async def _extract_single(self, url: dict) -> Optional[Dict[str, str]]:
        retry_count = 0
        max_retries = self._get_retries_for_domain(url["url"])
        last_error = None
        title = url["title"]
        original_url = url["url"]  # 원본 URL 저장

        while retry_count < max_retries:
            try:
                async with self.session.get(url["url"]) as response:
                    response.raise_for_status()
                    # 응답의 인코딩 확인
                    content_type = response.headers.get("content-type", "").lower()
                    if "charset=" in content_type:
                        encoding = content_type.split("charset=")[-1].strip()
                    else:
                        encoding = None

                    # 인코딩 시도 순서
                    encodings_to_try = [
                        encoding,  # 헤더에서 감지한 인코딩
                        "utf-8",  # 가장 일반적인 인코딩
                        "cp949",  # 한글 Windows
                        "euc-kr",  # 한글 Unix
                        "iso-8859-1",  # 기본 라틴
                        "utf-16",  # 유니코드
                        "utf-32",  # 유니코드
                    ]

                    html_content = None
                    for enc in encodings_to_try:
                        if not enc:
                            continue
                        try:
                            html_content = await response.text(encoding=enc)
                            logger.info(
                                f"[ArticleExtractor] 인코딩 성공: {enc} - URL: {url['url']}"
                            )
                            break
                        except UnicodeDecodeError:
                            continue

                    if not html_content:
                        logger.error(
                            f"[ArticleExtractor] 모든 인코딩 시도 실패 - URL: {url['url']}"
                        )
                        return {
                            "url": original_url,  # 원본 URL 반환
                            "title": title,
                            "input_text": "",
                        }

                text = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=False,
                    config=config,
                )

                if not text or not text.strip():
                    logger.warning(
                        f"[ArticleExtractor] 본문 추출 실패, 제목을 본문으로 사용 - URL: {url['url']}"
                    )
                    text = title

                return {
                    "url": original_url,  # 원본 URL 반환
                    "title": title,
                    "input_text": text.strip(),
                }

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retry_count += 1
                last_error = e
                if retry_count < max_retries:
                    wait_time = 2**retry_count
                    logger.info(
                        f"[ArticleExtractor] 재시도 대기 중 - URL: {url['url']}, "
                        f"시도: {retry_count}/{max_retries}, 대기 시간: {wait_time}초"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"[ArticleExtractor] URL 처리 실패 - {url['url']}, "
                        f"에러: {type(last_error).__name__}: {str(last_error)}, "
                        f"재시도: {retry_count}회"
                    )
                    return {
                        "url": original_url,  # 원본 URL 반환
                        "title": title,
                        "input_text": "",
                    }

            except Exception as e:
                logger.error(
                    f"[ArticleExtractor] 기사 추출 실패 - URL: {url['url']}, "
                    f"에러: {type(e).__name__}: {str(e)}"
                )
                return {
                    "url": original_url,  # 원본 URL 반환
                    "title": title,
                    "input_text": "",
                }

    async def extract_single(self, url: dict) -> Optional[Dict[str, str]]:
        try:
            return await self._extract_single(url)
        except Exception as e:
            logger.error(f"[ArticleExtractor] 기사 추출 실패: {e}")
            return None

    async def search(self, urls: List[dict]) -> AsyncGenerator[dict, None]:
        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_extract(url: dict) -> Optional[dict]:
            async with semaphore:
                try:
                    result = await self.extract_single(url)
                    if result:
                        return {"id": url.get("id", 0), **result}
                except Exception as e:
                    logger.error(f"[ArticleExtractor] 기사 추출 실패: {e}")
                return None

        tasks = [asyncio.create_task(bounded_extract(url)) for url in urls]

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    yield result
            except Exception as e:
                logger.error(f"[ArticleExtractor] 작업 처리 실패: {e}")
                continue


def cosine_similarity(a: List[float], b: List[float]) -> float:
    try:
        a, b = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception as e:
        logger.error(f"[cosine_similarity] 코사인 유사도 계산 중 오류 발생: {e}")
        return 0.0
