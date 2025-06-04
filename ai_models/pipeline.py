import asyncio
from typing import List, Dict
from ai_models.host import Host, SystemRole
from scrapers.article_extractor import ArticleExtractor
from ai_models.manager import BatchManager, wrapper
from ai_models.store import ResultStore
import logging
from utils.logger import Logger

logger = Logger.get_logger("ai_models.pipeline", log_level=logging.ERROR)


async def Pipeline(
    urls: List[Dict[str, str]],
    server: str,
    model: str,
    repeat: int = 1,
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
        repeat: 각 URL당 반복 처리 횟수
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not urls:
        logger.warning("[PIPELINE]: 입력 URL이 없습니다.")
        return {}

    logger.info(f"[PIPELINE]: {len(urls)}개 URL, {len(roles)}개 역할, 반복 {repeat}")

    results_dict = ResultStore()
    manager = None
    runner = None

    try:
        async with Host(server, model) as host:
            extractor = ArticleExtractor()

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
                    logger.warning(f"[PIPELINE] URL이 없는 항목 무시: {item}")

            # 텍스트 추출
            url_sentences = {}
            async for result in extractor.search(urls):
                if not result or not result.get("input_text"):
                    print(f"❌ 문장 분리 실패: URL={result['url']}")
                    continue
                url_sentences[result["url"]] = result["input_text"]
            # 태스크 생성
            tasks = []
            for url, input_text in url_sentences.items():
                for _ in range(repeat):
                    for role in roles:
                        logger.info(
                            f"[PIPELINE] URL: {url}, ROLE: {role}, TEXT: {input_text[:10]}..."
                        )
                        tasks.append(
                            asyncio.create_task(wrapper(url, role, input_text, manager))
                        )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            manager.running = False
            await asyncio.sleep(1.0)

            for task in results:
                if isinstance(task, Exception):
                    logger.error(f"[PIPELINE] 태스크 예외 발생: {repr(task)}")
                    continue
                url, role, response = task
                logger.info(f"[PIPELINE] URL : {url}")
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

            logger.info("[PIPELINE] 모든 태스크 처리 완료")

    except Exception as e:
        import traceback

        logger.error(f"[PIPELINE] 실행 중 예외 발생: {e}")
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
            except asyncio.TimeoutError as e:
                logger.warning((f"[PIPELINE] {e}"))
            except asyncio.CancelledError as e:
                logger.warning(f"[PIPELINE] {e}")

    return results_dict.as_dict()


async def TotalPipeline(
    text: list[str],
    server,
    model,
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

            # 태스크 결과 처리
            for task in asyncio.as_completed(tasks):
                try:
                    url, role, response = await task
                    content = (
                        response["choices"][0]["message"]["content"]
                        if response
                        else "실패"
                    )
                    results_dict.add_result(url=url, role=role, content=content)
                except Exception as e:
                    logger.error(f"[TotalPipeline] 태스크 처리 중 오류: {e}")
                    continue

            # 리소스 정리
            manager.running = False
            await asyncio.sleep(1.0)

    except Exception as e:
        logger.error(f"[TotalPipeline] 실행 중 예외 발생: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise
    finally:
        # 리소스 정리
        if manager:
            manager.running = False
            await asyncio.sleep(0.5)

        if runner and not runner.done():
            try:
                await asyncio.wait_for(runner, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                logger.warning(f"[TotalPipeline] Runner 정리 중 오류: {e}")

    return results_dict.as_dict()


if __name__ == "__main__":
    import time
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    SERVER = os.getenv("SERVER")
    MODEL = os.getenv("MODEL")
    # SERVER = "http://1d5d-34-124-161-59.ngrok-free.app"
    # MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    URLS = [
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192251.html",
            "title": '말 바꾼 윤석열 "계엄 길어야 하루"…헌재선 "며칠 예상"',
        },
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192255.html",
            "title": "윤석열 40분간 \"계엄은 평화적 메시지\"…판사도 발언 '시간조절' 당부",
        },
        {
            "url": "https://www.hankyung.com/article/2025041493977",
            "title": "'[속보] 韓대행 '국무위원들과 제게 부여된 마지막 소명 다할 것'",
        },
        {
            "url": "https://www.ytn.co.kr/_ln/0134_202505210904491383",
            "title": "다 꺼낸 구글의 '승부수'...삼성 이어 젠틀몬스터와 손 잡는다 [지금이뉴스]",
        },
        {
            "url": "https://m.news.zum.com/articles/98407004/%EC%9D%B4%EB%B2%88%EC%97%94-%EC%93%B8%EB%A7%8C%ED%95%A0%EA%B9%8C-%EA%B5%AC%EA%B8%80-%EC%82%BC%EC%84%B1-%EC%8A%A4%EB%A7%88%ED%8A%B8-%EC%95%88%EA%B2%BD-%ED%98%91%EC%97%85",
            "title": "이번엔 쓸만할까?…구글-삼성 스마트 안경 협업",
        },
        {
            "url": "https://www.news1.kr/world/usa-canada/5789241",
            "title": "챗봇에 흔들린 구글, AI 모드로 검색 기능 강화 나섰다",
        },
        {
            "url": "https://www.newstong.co.kr/view3.aspx?seq=13665995&allSeq=27&txtSearch=&cate=0&cnt=-5&subCate=2&order=default&newsNo=0",
            "title": "구글, 음성·영상으로도 검색 이용…예약 등 에이전트 기능도",
        },
    ]
    start = time.perf_counter()
    roles = [SystemRole.summary, SystemRole.tag, SystemRole.title]

    result1 = asyncio.run(Pipeline(URLS, SERVER, MODEL, repeat=1))
    from pprint import pprint

    print(result1)
    texts = [s for r in result1.values() for s in r.get("summary", [])]
    print(texts)
    result2 = asyncio.run(TotalPipeline(texts, SERVER, MODEL, repeat=1))
    print(result2)
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
