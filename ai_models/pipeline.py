import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from ai_models.host import Host, SystemRole
from scrapers.article_extractor import ArticleExtractor, ArticleFilter, ArticleParser
from ai_models.manager import BatchManager, wrapper
from ai_models.store import ResultStore
import logging
from utils.logger import Logger
from line_profiler import profile

logger = Logger.get_logger("ai_models.pipeline", log_level=logging.ERROR)


@profile
async def Pipeline(
    urls: List[Dict[str, str]],
    server: str,
    model: str,
    repeat: int = 1,
    roles: List[SystemRole] = None,
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
    import dotenv
    import os

    dotenv.load_dotenv(override=True)
    # 테스트 설정
    SERVER = os.getenv("SERVER")  # 실제 서버 URL로 변경 필요
    MODEL = os.getenv("MODEL")  # 실제 모델 이름으로 변경 필요

    # 테스트 URL
    URLS = [
        {
            "url": "https://www.christiandaily.co.kr/news/147620",
            "title": "내년 최저임금 논의 본격화… 노동계 “인상”, 경영계 “동결” 팽팽한 대립",
        },
        {
            "url": "https://www.sisajournal.com/news/articleView.html?idxno=334605",
            "title": "대구기업 절반 이상 “현 최저임금 높다”···내년 '동결 또는 1%미만 인상' 선호",
        },
        {
            "url": "https://www.newscj.com/news/articleView.html?idxno=3273593",
            "title": "내년 최저임금 논의 본격화… 생계비냐 수용성이냐",
        },
        {
            "url": "https://worknworld.kctu.org/news/articleView.html?idxno=507086",
            "title": '차기 정부 최저임금 인상 목표 묻자 ··· 민주당 "노사 협한 ...',
        },
        {
            "url": "https://www.pn.or.kr/news/articleView.html?idxno=31985",
            "title": '단속에 다치고, 최저임금도 못 받아...미등록 이주노동자들 "폭력단속 ...',
        },
        {
            "url": "https://www.business-humanrights.org/ko/latest-news/s-korea-government-plans-controversial-pilot-scheme-for-foreign-domestic-workers-without-minimum-wage-protection/",
            "title": "한국: 정부, 최저임금 보호 없는 외국인 가사 노동자 시행 시범기획",
        },
        {
            "url": "https://www.ablenews.co.kr/news/articleView.html?idxno=213776",
            "title": "2025년 최저임금 시급 '1만 30원', 월급 '209만 6270원'",
        },
        {
            "url": "https://www.sisain.co.kr/news/articleView.html?idxno=55089",
            "title": "최저선 무너진 인권위, 어쩌다 이 지경에",
        },
        {
            "url": "https://www.yna.co.kr/view/AKR20250527119900530",
            "title": "본격화한 내년 최저임금 '기싸움'…“대폭 인상” vs “위기 상황”",
        },
        {
            "url": "https://www.khan.co.kr/article/202505271636001",
            "title": "“최저임금 적용 확대” VS “업종별 차등 적용”···최임위 기싸움 본격화",
        },
        {
            "url": "https://www.newsis.com/view/NISX20250527_0003191864",
            "title": "최저임금위 2차 회의…노동계 '확대적용' vs 경영계 '차등적용'",
        },
        {
            "url": "https://www.sukbakmagazine.com/news/articleView.html?idxno=64660",
            "title": "소공연, “역대급 위기, 내년 최저임금 동결해야”",
        },
        {
            "url": "https://news.kbs.co.kr/news/view.do?ncd=8264706",
            "title": "최저임금위원회 2차 전원회의…“대폭 인상” vs “위기 상황”",
        },
        {
            "url": "https://news.nate.com/view/20250527n01337",
            "title": "오늘 최저임금위 2차 회의…“아직 부족” vs “그동안 너무 올라”",
        },
        {
            "url": "https://news.nate.com/view/20250526n08037",
            "title": "최저임금 토론회 개최…한국노총 '새 정부 노동정책 첫 지표는 최저임금'",
        },
        {
            "url": "https://v.daum.net/v/20250527161200184",
            "title": "본격화한 내년 최저임금 '기싸움'…“대폭 인상” vs “위기 상황”",
        },
        {
            "url": "https://www.christiandaily.co.kr/news/147620",
            "title": "내년 최저임금 논의 본격화… 노동계 “인상”, 경영계 “동결” 팽팽한 대립",
        },
        {
            "url": "https://www.sisajournal.com/news/articleView.html?idxno=334605",
            "title": "대구기업 절반 이상 “현 최저임금 높다”···내년 '동결 또는 1%미만 인상' 선호",
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
