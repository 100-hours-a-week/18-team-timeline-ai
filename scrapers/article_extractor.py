import aiohttp
import asyncio
import numpy as np
import trafilatura
from trafilatura.settings import use_config
from utils.logger import Logger
from utils.timeline_utils import contains_korean
from config.settings import (
    USER_AGENT,
    ARTICLE_TIMEOUT,
    MAX_WORKERS,
    DEFAULT_LANG,
    DOMAIN_TIMEOUTS,
    DOMAIN_RETRIES,
    CLIENT_TIMEOUT,
)
from typing import List, Dict, Optional, AsyncGenerator
from scrapers.base_searcher import BaseSearcher
from urllib.parse import urlparse

logger = Logger.get_logger("article_extractor")
config = use_config()
config.set("DEFAULT", "user-agent", USER_AGENT)
headers = {"User-Agent": USER_AGENT}


class ArticleExtractor(BaseSearcher):
    def __init__(self, lang: str = DEFAULT_LANG, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self.lang = lang
        self.session = None
        self.domain_timeouts = DOMAIN_TIMEOUTS
        self.domain_retries = DOMAIN_RETRIES
        logger.info(
            f"[ArticleExtractor] 초기화 완료 - 언어: {lang}, "
            f"최대 작업자 수: {max_workers}"
        )

    async def __aenter__(self):
        logger.info("[ArticleExtractor] 세션 생성 시작")
        timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            connector=aiohttp.TCPConnector(
                limit=self.max_workers, force_close=True, enable_cleanup_closed=True
            ),
        )
        logger.info(
            f"[ArticleExtractor] 세션 생성 완료 - "
            f"작업자 수: {self.max_workers}, 타임아웃: {CLIENT_TIMEOUT}초"
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.session.closed:
            logger.info("[ArticleExtractor] 세션 종료 시작")
            await self.session.close()
            logger.info("[ArticleExtractor] 세션 종료 완료")

    def _get_timeout_for_domain(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            timeout = self.domain_timeouts.get(domain, self.domain_timeouts["default"])
            logger.debug(
                f"[ArticleExtractor] 도메인 타임아웃 설정 - "
                f"도메인: {domain}, 타임아웃: {timeout}초"
            )
            return timeout
        except Exception as e:
            logger.warning(
                f"[ArticleExtractor] 도메인 타임아웃 설정 실패 - "
                f"URL: {url}, 기본값 사용: {self.domain_timeouts['default']}초"
            )
            return self.domain_timeouts["default"]

    def _get_retries_for_domain(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc
            retries = self.domain_retries.get(domain, self.domain_retries["default"])
            logger.debug(
                f"[ArticleExtractor] 도메인 재시도 설정 - "
                f"도메인: {domain}, 재시도 횟수: {retries}회"
            )
            return retries
        except Exception as e:
            logger.warning(
                f"[ArticleExtractor] 도메인 재시도 설정 실패 - "
                f"URL: {url}, 기본값 사용: {self.domain_retries['default']}회"
            )
            return self.domain_retries["default"]

    async def _extract_single(self, url: dict) -> Optional[Dict[str, str]]:
        retry_count = 0
        max_retries = self._get_retries_for_domain(url["url"])
        last_error = None
        title = url["title"]
        original_url = url["url"]

        logger.info(
            f"[ArticleExtractor] 기사 추출 시작 - "
            f"URL: {original_url}, 제목: {title}, "
            f"최대 재시도: {max_retries}회"
        )

        while retry_count < max_retries:
            try:
                async with self.session.get(url["url"]) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()

                    logger.debug(
                        f"[ArticleExtractor] 응답 헤더 확인 - "
                        f"상태 코드: {response.status}, "
                        f"Content-Type: {content_type}"
                    )

                    if "charset=" in content_type:
                        encoding = content_type.split("charset=")[-1].strip()
                    else:
                        encoding = None

                    encodings_to_try = [
                        encoding,
                        "utf-8",
                        "cp949",
                        "euc-kr",
                        "iso-8859-1",
                        "utf-16",
                        "utf-32",
                    ]

                    html_content = None
                    for enc in encodings_to_try:
                        if not enc:
                            continue
                        try:
                            html_content = await response.text(encoding=enc)
                            logger.info(
                                f"[ArticleExtractor] 인코딩 성공 - "
                                f"URL: {url['url']}, 인코딩: {enc}"
                            )
                            break
                        except UnicodeDecodeError:
                            logger.debug(
                                f"[ArticleExtractor] 인코딩 실패 - "
                                f"URL: {url['url']}, 인코딩: {enc}"
                            )
                            continue

                    if not html_content:
                        logger.error(
                            f"[ArticleExtractor] 모든 인코딩 시도 실패 - "
                            f"URL: {url['url']}"
                        )
                        return {
                            "url": original_url,
                            "title": title,
                            "input_text": "",
                        }

                text = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=False,
                    config=config,
                )

                if (not text) or (not text.strip()) or (len(text) < 200):
                    logger.warning(
                        f"[ArticleExtractor] 본문이 없습니다 - "
                        f"URL: {url['url']}, 제목을 본문으로 사용"
                    )
                    text = title

                if not contains_korean(text):
                    logger.error(
                        "[ArticleExtractor] 본문에 한글이 없어 건너뜁니다. - "
                        f"url: {original_url} - "
                    )
                    text = ""

                logger.info(
                    f"[ArticleExtractor] 기사 추출 완료 - "
                    f"URL: {url['url']}, 본문 길이: {len(text)}자"
                )

                return {
                    "url": original_url,
                    "title": title,
                    "input_text": text.strip(),
                }

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retry_count += 1
                last_error = e
                if retry_count < max_retries:
                    wait_time = 2**retry_count
                    logger.info(
                        f"[ArticleExtractor] 재시도 대기 - "
                        f"URL: {url['url']}, "
                        f"시도: {retry_count}/{max_retries}, "
                        f"대기 시간: {wait_time}초"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"[ArticleExtractor] URL 처리 실패 - "
                        f"URL: {url['url']}, "
                        f"에러: {type(last_error).__name__}: {str(last_error)}, "
                        f"재시도: {retry_count}회"
                    )
                    return {
                        "url": original_url,
                        "title": title,
                        "input_text": "",
                    }

            except Exception as e:
                logger.error(
                    f"[ArticleExtractor] 기사 추출 실패 - "
                    f"URL: {url['url']}, "
                    f"에러: {type(e).__name__}: {str(e)}"
                )
                return {
                    "url": original_url,
                    "title": title,
                    "input_text": "",
                }

    async def extract_single(self, url: dict) -> Optional[Dict[str, str]]:
        try:
            return await self._extract_single(url)
        except Exception as e:
            logger.error(
                f"[ArticleExtractor] 기사 추출 실패 - "
                f"URL: {url.get('url', 'unknown')}, "
                f"에러: {str(e)}"
            )
            return None

    async def search(self, urls: List[dict]) -> AsyncGenerator[dict, None]:
        logger.info(
            f"[ArticleExtractor] 배치 처리 시작 - "
            f"URL 수: {len(urls)}, 최대 작업자 수: {self.max_workers}"
        )

        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_extract(url: dict) -> Optional[dict]:
            async with semaphore:
                try:
                    result = await self.extract_single(url)
                    if result:
                        return {"id": url.get("id", 0), **result}
                except Exception as e:
                    logger.error(
                        f"[ArticleExtractor] 기사 추출 실패 - "
                        f"URL: {url.get('url', 'unknown')}, "
                        f"에러: {str(e)}"
                    )
                return None

        tasks = [asyncio.create_task(bounded_extract(url)) for url in urls]
        completed = 0

        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                completed += 1
                if result:
                    logger.info(
                        f"[ArticleExtractor] 작업 완료 - "
                        f"진행: {completed}/{len(urls)}, "
                        f"URL: {result.get('url', 'unknown')}"
                    )
                    yield result
            except Exception as e:
                logger.error(
                    f"[ArticleExtractor] 작업 처리 실패 - "
                    f"진행: {completed}/{len(urls)}, "
                    f"에러: {str(e)}"
                )
                continue

        logger.info(
            f"[ArticleExtractor] 배치 처리 완료 - "
            f"총 URL 수: {len(urls)}, 완료: {completed}"
        )
