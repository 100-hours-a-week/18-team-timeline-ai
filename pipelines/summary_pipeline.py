from typing import List, Dict
from config.prompts import SystemRole
from scrapers.article_extractor import ArticleExtractor
from inference.host import Host
from inference.manager import BatchManager, wrapper
from utils.store import ResultStore
from utils.url_utils import clean_url
import logging
from utils.logger import Logger
import asyncio

logger = Logger.get_logger("ai_models.pipeline", log_level=logging.ERROR)


async def Pipeline(
    urls: List[Dict[str, str]],
    server: str,
    model: str,
    roles: List[SystemRole] = [SystemRole.summary],
    batch_size: int = 256,
    max_wait_time: float = 2.0,
) -> Dict[str, Dict[str, List[str]]]:
    """
    URL 목록에 대한 AI 처리 파이프라인

    Args:
        urls: 처리할 URL 목록
        server: AI 서버 URL
        model: 사용할 모델 이름
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not urls:
        logger.warning("[PIPELINE]: 입력 URL이 없습니다.")
        return {}

    logger.info(f"[PIPELINE]: {len(urls)}개 URL, {len(roles)}개 역할")

    # URL 정리
    cleaned_urls = []
    for url_dict in urls:
        cleaned_url = clean_url(url_dict["url"])
        cleaned_urls.append(
            {
                "url": cleaned_url,
                "title": url_dict["title"],
                "id": url_dict.get("id", 0),
            }
        )

    results_dict = ResultStore()
    manager = None
    runner = None

    try:
        async with Host(server, model) as host:
            # 텍스트 추출
            url_sentences = {}
            async with ArticleExtractor() as extractor:
                async for result in extractor.search(cleaned_urls):
                    if not result:
                        logger.warning(f"[SummaryPipeline] 결과가 없는 URL 건너뜀")
                        continue

                    if not result.get("input_text"):
                        logger.warning(
                            f"[SummaryPipeline] 본문 추출 실패: URL={result.get('url', 'unknown')}, "
                            f"제목={result.get('title', 'unknown')}"
                        )
                        continue

                    url_sentences[result["url"]] = result["input_text"]
                    logger.info(
                        f"[SummaryPipeline] 본문 추출 성공: URL={result['url']}, "
                        f"제목={result['title']}, "
                        f"본문 길이={len(result['input_text'])}"
                    )

            # 배치 매니저 초기화
            manager = BatchManager(
                host, batch_size=batch_size, max_wait_time=max_wait_time
            )
            runner = asyncio.create_task(manager.run())

            # 태스크 생성
            tasks = []
            for url, input_text in url_sentences.items():
                for role in roles:
                    logger.info(
                        f"[PIPELINE] URL: {url}, ROLE: {role}, "
                        f"TEXT: {input_text[:10]}..."
                    )
                    tasks.append(
                        asyncio.create_task(wrapper(url, role, input_text, manager))
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # 리소스 정리
        if manager:
            manager.running = False
            await asyncio.sleep(1.0)  # 대기 시간 증가

        if runner:
            try:
                await asyncio.wait_for(runner, timeout=5.0)  # 타임아웃 증가
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                logger.warning(f"[PIPELINE] Runner 태스크 정리 중 오류: {e}")
                if not runner.done():
                    runner.cancel()
                    try:
                        await runner
                    except asyncio.CancelledError:
                        pass

    for task in results:
        if isinstance(task, Exception):
            logger.error(f"[PIPELINE] 태스크 예외 발생: {repr(task)}")
            continue
        url, role, response = task
        logger.info(f"[PIPELINE] URL : {url}")
        if isinstance(response, dict) and "error" in response:
            logger.error(
                f"응답 오류: URL={url}, ROLE={role}, " f"ERROR={response['error']}"
            )
            continue

        try:
            content = (
                response["choices"][0]["message"]["content"] if response else "실패"
            )
            results_dict.add_result(url=url, role=role, content=content)
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"응답 파싱 오류: {e}, 응답: {response}")

    logger.info("[PIPELINE] 모든 태스크 처리 완료")

    return results_dict.as_dict()
