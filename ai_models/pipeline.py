import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from ai_models.host import Host, SystemRole
from scrapers.article_extractor import ArticleExtractor, ArticleFilter, ArticleParser
from ai_models.manager import BatchManager, wrapper
from ai_models.store import ResultStore
import logging
from utils.logger import Logger

logger = Logger.get_logger("ai_models.pipeline", log_level=logging.ERROR)


async def Pipeline(
    urls: List[Dict[str, str]],
    server: str,
    model: str,
    repeat: int = 3,
    roles: List[SystemRole] = None,
    batch_size: int = 64,
    max_wait_time: float = 2.0,
) -> Dict[str, Dict[str, List[str]]]:
    """
    URL 목록에 대한 AI 처리 파이프라인

    Args:
        urls: 처리할 URL 목록
        server: AI 서버 URL
        model: 사용할 모델 이름
        repeat: 각 URL당 반복 처리 횟수
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not urls:
        logger.warning("Pipeline: 입력 URL이 없습니다.")
        return {}

    if roles is None:
        roles = [SystemRole.SUMMARY]

    logger.info(
        f"Pipeline 시작: {len(urls)}개 URL, {len(roles)}개 역할, 반복 {repeat}회"
    )

    results_dict = ResultStore()
    manager = None
    runner = None

    try:
        async with Host(server, model) as host:
            extractor = ArticleExtractor()
            parser = ArticleParser()
            filter = ArticleFilter()

            manager = BatchManager(
                host, batch_size=batch_size, max_wait_time=max_wait_time
            )
            runner = asyncio.create_task(manager.run())

            # URL 등록
            for item in urls:
                url = item.get("url")
                if url:
                    results_dict.register({"url": url})
                else:
                    logger.warning(f"URL이 없는 항목 무시: {item}")

            # 텍스트 추출
            url_sentences = {}
            """
            async with filter:
                async for result in extractor.search(urls):
                    parsed_list = await parser.parse(result)
                    key_sentences = await filter.extract_key_sentences(parsed_list)

                    url_sentences[result["url"]] = "\n".join(key_sentences)
            """

            async for result in extractor.search(urls):
                url_sentences[result["url"]] = result["input_text"]
            # 태스크 생성
            tasks = []
            for url, input_text in url_sentences.items():
                logger.debug(f"URL 처리 중: {url}, 텍스트 길이: {len(input_text)}")
                for _ in range(repeat):
                    for role in roles:
                        logger.info(
                            f"[PIPELINE] URL: {url}, ROLE: {role}, 텍스트: {input_text[:30]}..."
                        )
                        tasks.append(
                            asyncio.create_task(wrapper(url, role, input_text, manager))
                        )

            # 태스크 실행 및 결과 수집
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"태스크 실행 중 예외 발생: {result}")
                    continue

                url, role, response = result

                if isinstance(response, dict) and "error" in response:
                    logger.error(
                        f"응답 오류: URL={url}, ROLE={role}, ERROR={response['error']}"
                    )
                    continue

                try:
                    content = (
                        response["choices"][0]["message"]["content"]
                        if response
                        else "실패"
                    )
                    results_dict.add_result(url=url, role=role, content=content)
                except (KeyError, TypeError, IndexError) as e:
                    logger.error(f"응답 파싱 오류: {e}, 응답: {response}")

            logger.info("모든 태스크 처리 완료")
            if logger.log_level <= logging.INFO:
                results_dict.display()

    except Exception as e:
        import traceback

        logger.error(f"Pipeline 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # 리소스 정리
        if manager:
            manager.running = False
            await asyncio.sleep(0.5)

        if runner:
            try:
                await asyncio.wait_for(runner, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Runner 태스크 정리 중 타임아웃 또는 취소됨")

    return results_dict.as_dict()


async def TotalPipeline(
    text: List[str],
    server: str,
    model: str,
    repeat: int = 5,
    roles: list[SystemRole] = [SystemRole.summary, SystemRole.title],
    batch_size: int = 5,
    max_wait_time: float = 0.5,
) -> Dict[str, Dict[str, List[str]]]:
    """
    텍스트 목록에 대한 통합 AI 처리 파이프라인

    Args:
        text: 처리할 텍스트 목록
        server: AI 서버 URL
        model: 사용할 모델 이름
        repeat: 각 텍스트당 반복 처리 횟수
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not text:
        logger.warning("TotalPipeline: 입력 텍스트가 없습니다.")
        return {}

    if roles is None:
        roles = [SystemRole.SUMMARY, SystemRole.TITLE, SystemRole.TAG]

    logger.info(
        f"TotalPipeline 시작: {len(text)}개 텍스트, {len(roles)}개 역할, 반복 {repeat}회"
    )

    results_dict = ResultStore()
    manager = None
    runner = None

    try:
        async with Host(server, model) as host:
            results_dict.register({"url": "total_summary"})
            manager = BatchManager(
                host=host, batch_size=batch_size, max_wait_time=max_wait_time
            )
            runner = asyncio.create_task(manager.run())

            # 텍스트 결합
            sentence = ". ".join(text)
            logger.debug(f"결합된 텍스트 길이: {len(sentence)}")

            # 태스크 생성 및 실행
            tasks = []
            for _ in range(repeat):
                for role in roles:
                    tasks.append(
                        asyncio.create_task(
                            wrapper("total_summary", role, sentence, manager)
                        )
                    )

            # 결과 수집 (완료되는 순서대로)
            for task in asyncio.as_completed(tasks):
                try:
                    url, role, response = await task

                    if isinstance(response, dict) and "error" in response:
                        logger.error(
                            f"응답 오류: URL={url}, ROLE={role}, ERROR={response['error']}"
                        )
                        continue

                    content = (
                        response["choices"][0]["message"]["content"]
                        if response
                        else "실패"
                    )
                    results_dict.add_result(url=url, role=role, content=content)
                except Exception as e:
                    logger.error(f"태스크 처리 중 예외 발생: {e}")

            logger.info("모든 태스크 처리 완료")

    except Exception as e:
        import traceback

        logger.error(f"TotalPipeline 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # 리소스 정리
        if manager:
            manager.running = False
            await asyncio.sleep(0.5)

        if runner:
            try:
                await asyncio.wait_for(runner, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning("Runner 태스크 정리 중 타임아웃 또는 취소됨")

    return results_dict.as_dict()


if __name__ == "__main__":
    import time
    from pprint import pprint

    import os

    # 테스트 설정
    SERVER = os.getenv("SERVER")  # 실제 서버 URL로 변경 필요
    MODEL = os.getenv("MODEL")  # 실제 모델 이름으로 변경 필요

    # 테스트 URL
    URLS = [
        {
            "url": "https://www.example.com/article1",
            "title": "테스트 기사 1",
        },
        {
            "url": "https://www.example.com/article2",
            "title": "테스트 기사 2",
        },
    ]

    # 실행 시간 측정
    start = time.perf_counter()

    # 역할 설정
    roles = [SystemRole.SUMMARY, SystemRole.TAG, SystemRole.TITLE]

    # 파이프라인 실행
    try:
        # 첫 번째 파이프라인 실행
        result1 = asyncio.run(Pipeline(URLS, SERVER, MODEL, repeat=1, roles=roles))
        print("Pipeline 결과:")
        pprint(result1)

        # 결과에서 요약 텍스트 추출
        texts = [s for r in result1.values() for s in r.get("summary", [])]
        print(f"\n추출된 요약 텍스트: {len(texts)}개")

        # 두 번째 파이프라인 실행
        result2 = asyncio.run(
            TotalPipeline(texts, SERVER, MODEL, repeat=1, roles=roles)
        )
        print("\nTotalPipeline 결과:")
        pprint(result2)
    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")

    # 실행 시간 출력
    end = time.perf_counter()
    print(f"\n총 실행 시간: {end - start:.2f}s")
