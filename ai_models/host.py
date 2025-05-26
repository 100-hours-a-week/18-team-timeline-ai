import aiohttp
from utils.logger import Logger
from utils.handling import handle_http_error
from enum import Enum, auto
import logging
import orjson
from typing import Dict, Any, Optional, Union
import asyncio
import tiktoken

logger = Logger.get_logger("ai_models.host", log_level=logging.ERROR)


class SystemRole(Enum):
    """시스템 역할 정의 - 모든 파일에서 일관되게 사용하기 위한 열거형"""

    SUMMARY = "summary"
    TITLE = "title"
    TAG = "tag"

    def __str__(self) -> str:
        """문자열 표현을 위한 메서드"""
        return self.value


# 시스템 프롬프트를 별도 파일이나 설정으로 분리하는 것이 좋지만, 현재는 코드 내에 유지
SYSTEM_PROMPT = {
    SystemRole.SUMMARY: """
            You are a world-class summarization AI specialized in news content.
            You take long-form news articles and return only a concise, 1–2 sentence summary of the most important information.
            IMPORTANT RULES:
            Do not include any commentary, category label, preface, explanation, or formatting.
            Only output the summary. Nothing else.
            Never say things like "Here is the summary" or "This article is about..."
            If the input text is not a valid article, just return: "Invalid input".
            Now, summarize the following article strictly following the above rules:
        """,
    SystemRole.TITLE: """
            You are a high-performance summarization and headline-generation AI specialized in news content.
            You will read a full news article and return only two things:
            A short, attention-grabbing headline (under 12 words)
            A 1–2 sentence summary that clearly conveys the core of the article
            STRICT RULES:
            Do not write anything else. No labels, no explanation, no markdown.
        """,
    SystemRole.TAG: """
                You are a Korean news classification AI.  
                Read the full article and determine its category.
                **출력 규칙**:
                - 아래 다섯 가지 중 **가장 적합한 태그 하나만 출력**:
                - 경제
                - 연예
                - 스포츠
                - 과학
                - 기타
                - **한국어로만 출력**
                - **그 외 텍스트 절대 금지**
                - 뉴스가 아닌 경우 출력: `입력 오류`
                다음 기사를 분류하세요:
        """,
}

# 모델별 인코딩 매핑
MODEL_ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct": "cl100k_base",  # 예시, 실제 모델에 맞게 조정 필요
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B": "cl100k_base",  # 예시, 실제 모델에 맞게 조정 필요
    "default": "cl100k_base",  # 기본값
}


def truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    model: str = "default",
    add_truncation_notice: bool = True,
) -> str:
    """
    텍스트를 지정된 최대 토큰 수로 제한합니다.

    Args:
        text: 제한할 텍스트
        max_tokens: 최대 토큰 수
        model: 토큰화에 사용할 모델 이름
        add_truncation_notice: 잘린 경우 알림 추가 여부

    Returns:
        str: 토큰 수로 제한된 텍스트
    """
    if not text:
        return text

    # 모델에 맞는 인코딩 선택
    encoding_name = MODEL_ENCODINGS.get(model, MODEL_ENCODINGS["default"])

    try:
        # 토크나이저 가져오기
        encoding = tiktoken.get_encoding(encoding_name)

        # 텍스트를 토큰으로 인코딩
        tokens = encoding.encode(text)

        # 토큰 수 확인
        if len(tokens) <= max_tokens:
            return text

        # 토큰 제한
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)

        # 잘렸다는 알림 추가 (선택적)
        if add_truncation_notice:
            truncated_text += f"\n\n[참고: 입력 텍스트가 {len(tokens)}개 토큰으로 너무 길어 {max_tokens}개 토큰으로 제한되었습니다.]"

        logger.info(
            f"텍스트가 {len(tokens)}개 토큰에서 {max_tokens}개 토큰으로 제한되었습니다."
        )
        return truncated_text

    except Exception as e:
        logger.warning(f"토큰 제한 중 오류 발생: {e}. 원본 텍스트를 반환합니다.")
        return text


class Host:
    """AI 모델 호스트와의 통신을 관리하는 클래스"""

    def __init__(
        self,
        host: str,
        model: str,
        timeout: int = 60,
        temperature: float = 0.5,
        max_tokens: int = 64,
        max_input_tokens: int = 2048,  # 입력 텍스트 최대 토큰 수 추가
        verbose: bool = False,
        concurrency: int = 256,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        초기화

        Args:
            host: 호스트 URL
            model: 모델 이름
            timeout: 요청 타임아웃(초)
            temperature: 모델 온도 파라미터
            max_tokens: 최대 출력 토큰 수
            max_input_tokens: 최대 입력 토큰 수
            verbose: 상세 로깅 여부
            concurrency: 동시 요청 수
            retry_attempts: 재시도 횟수
            retry_delay: 재시도 간 지연 시간(초)
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.session: Optional[aiohttp.ClientSession] = None
        self.verbose = verbose
        self.concurrency = concurrency
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._is_connected = False

    async def __aenter__(self):
        """
        비동기 컨텍스트 관리자 진입

        Raises:
            RuntimeError: 호스트 연결 실패

        Returns:
            Host: 호스트 객체
        """
        try:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self.concurrency)
            )
            if not await self.check_connection():
                logger.error(f"[Host] Failed to connect to the host: {self.host}")
                raise RuntimeError(f"Failed to connect to the host: {self.host}")

            self._is_connected = True
            logger.info(f"[Host] Connected to the host: {self.host}")
            return self
        except Exception as e:
            # 세션 생성 후 연결 실패 시 세션 정리
            if self.session is not None:
                await self.session.close()
                self.session = None
            raise RuntimeError(f"Connection initialization failed: {str(e)}")

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        비동기 컨텍스트 관리자 종료

        Args:
            exc_type: 예외 타입
            exc_value: 예외 값
            traceback: 예외 추적 정보
        """
        logger.info(f"[Host] Closing the host: {self.host}")
        await self.close()

    async def close(self):
        """
        세션 종료 및 리소스 정리
        """
        if self.session:
            await self.session.close()
            self.session = None
            self._is_connected = False
            logger.debug("[Host] Session closed")

    async def check_connection(self) -> bool:
        """
        호스트 연결 확인

        Returns:
            bool: 연결 성공 여부
        """
        if not self.session:
            logger.error("[Host] No active session")
            return False

        url = f"{self.host}/health"
        logger.info(f"[Host] Checking connection to the host: {url}")

        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                if self.verbose:
                    logger.info(f"[Host] Connection status: {response.status}")

                return response.status == 200
        except (
            aiohttp.ClientError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientResponseError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        ) as e:
            logger.error(f"[Host] Connection check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"[Host] Unexpected error during connection check: {e}")
            return False

    def count_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수를 계산합니다.

        Args:
            text: 토큰 수를 계산할 텍스트

        Returns:
            int: 토큰 수
        """
        encoding_name = MODEL_ENCODINGS.get(self.model, MODEL_ENCODINGS["default"])
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"토큰 계산 중 오류 발생: {e}")
            # 대략적인 추정: 영어 기준 평균 단어당 1.3 토큰
            return int(len(text.split()) * 1.3)

    async def query(self, task: SystemRole, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 모델에 쿼리 요청

        Args:
            task: 요청 타입
            payload: 요청 데이터

        Raises:
            RuntimeError: 세션이 초기화되지 않은 경우
            ValueError: 잘못된 요청 데이터
            Exception: 요청 실패

        Returns:
            Dict[str, Any]: 응답 데이터
        """
        if not self.session or not self._is_connected:
            raise RuntimeError("Host session is not initialized or connected")

        if not payload or "text" not in payload:
            raise ValueError("Invalid payload: must contain 'text' field")

        # 입력 텍스트를 토큰 수로 제한
        original_text = payload["text"]
        truncated_text = truncate_text_by_tokens(
            original_text, self.max_input_tokens, self.model
        )

        # 토큰 수 로깅
        if self.verbose and original_text != truncated_text:
            original_tokens = self.count_tokens(original_text)
            truncated_tokens = self.count_tokens(truncated_text)
            logger.info(
                f"[Host] 입력 텍스트가 {original_tokens}개 토큰에서 {truncated_tokens}개 토큰으로 제한되었습니다."
            )

        headers = {"Content-Type": "application/json"}

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT[task],
                },
                {"role": "user", "content": truncated_text},  # 제한된 텍스트 사용
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self.verbose:
            logger.debug(f"[Host] Query body: {body}")

        url = f"{self.host}/v1/chat/completions"

        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with self.session.post(
                    url,
                    data=orjson.dumps(body),
                    timeout=self.timeout,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    response_text = await response.text()

                    try:
                        result = orjson.loads(response_text)
                    except (ValueError, orjson.JSONDecodeError) as e:
                        logger.error(
                            f"[Host] JSON decode error: {e}, Response: {response_text[:100]}"
                        )
                        raise ValueError(f"Failed to parse response: {str(e)}")

                    success = await handle_http_error(result, body, logger)
                    if success:
                        if self.verbose:
                            logger.debug(f"[Host] Query successful: {result}")
                        return result
                    else:
                        logger.error(f"[Host] Query failed: {result}")
                        raise Exception(f"API error: {result}")

            except (
                aiohttp.ClientError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientResponseError,
                aiohttp.ServerDisconnectedError,
                asyncio.TimeoutError,
            ) as e:
                logger.warning(f"[Host] Query attempt {attempt} failed: {e}")

                if attempt < self.retry_attempts:
                    await asyncio.sleep(self.retry_delay * attempt)  # 지수 백오프
                else:
                    logger.error(f"[Host] All query attempts failed: {e}")
                    raise e
            except Exception as e:
                logger.error(f"[Host] Unexpected error during query: {e}")
                raise e

        # 모든 재시도 실패 시
        raise RuntimeError(f"All {self.retry_attempts} query attempts failed")


if __name__ == "__main__":
    import asyncio
    import time

    ret = []
    NUM_REQUESTS = 100  # 요청 수 조절 가능
    CONCURRENCY = 10  # 동시에 몇 개씩 실행할지 (이벤트 루프 병렬도)
    TEXT = """
        글로벌 배터리 시장의 38%를 점유하는 '대어' CATL이 홍콩 주식시장에 신규 상장했다. 상장 첫날 CATL 주가는 공모가보다 최대 18% 높은 가격에 거래됐다.
        국내 개인투자자들은 이번 기업공개(IPO)로 인해 CATL을 직접투자할 수 있는 길이 열렸다.
        20일(현지시간) 홍콩증권거래소에서 CATL은 장 초반 공모가인 263홍콩달러(약 4만6800원)보다 약 12.5% 높은 296홍콩달러(약 5만2700원)에 거래됐다.
        CATL은 이날 정오께 311.4달러(약 5만5400원)에 거래되며 최고가를 기록했다. 시초가 대비로는 5.2%, 공모가 대비로는 18.4% 높은 가격이다.
        오후 3시께 CATL은 307.6홍콩달러(약 5만4700원)에 거래됐다.
        CATL은 이번 IPO를 통해 46억달러(약 6조4000억원) 이상을 조달한 것으로 전해진다. 초과 배정 옵션을 행사할 경우 총 조달액은 53억달러(약 7조3000억원)까지 불어날 수 있다.
        이는 올해 전 세계 IPO 시장에서 최대 규모 금액이다. 지난해 홍콩증시에 상장했던 중국의 가전업체 메이디(46억달러)의 사례도 뛰어넘는다.
        2021년에 62억달러를 조달했던 중국의 온라인 플랫폼 기업 콰이쇼우테크놀로지와도 비견된다.
        CATL은 조달 금액의 90% 이상을 헝가리 공장 건설에 사용할 계획이다. 2027년까지 완공 예정인 이번 프로젝트를 통해 CATL은 유럽시장을 더욱 확장할 전망이다.
        이번 IPO 과정에선 중국석유화공(시노펙)과 쿠웨이트투자청, 카타르투자청, 힐하우스인베스트먼트, 오크트리캐피털 등이 주요 투자자로 참여했다.
        공모청약의 1억2540만주는 기관 투자자에게, 1016만주는 홍콩 개인 투자자에게 매각됐다. 이 과정에서 미국 개인투자자의 공모 참여를 제한하는 '레귤레이션 S' 방식이 활용되기도 했다.
        IPO 주관사는 중국국제금융공사(CICC)와 더불어 뱅크오브아메리카, 골드만삭스, 모건스탠리, JP모건등이 맡았다.
        CATL이 홍콩증시에 입성하면서 국내 개인투자자들에게도 CATL 직접투자의 길이 열렸다.
        CATL은 지난 2018년 중국 본토 선전증권거래소에 상장했지만, 이는 선전증시와 홍콩증시를 잇는 '선강퉁 제도'에 포함돼지 않아 외국인 개인투자자들의 매수가 사실상 불가능했다.
        이에 국내 개인투자자는 CATL이 포함된 상장지수펀드(ETF)를 매수하는 것이 최선이었다. 그러나 이번 상장을 통해 직접 매매가 가능해졌다.
        존슨 완 제프리스 중국 연구원은 이날 "CATL은 견조한 실적과 매력적 밸류에이션이 있어 앞으로 50% 이상 상승할 수 있다"며 CATL의 주가 성장 가능성을 높게 점쳤다.
        중국의 '배터리 굴기'를 대표하는 CATL은 이미 글로벌 배터리 산업을 주도하고 있다.
        SNE리서치에 따르면 CATL은 올해 1분기 기준으로 글로벌 배터리시장의 38.3%에 해당하는 84.9기가와트시(GWh)를 공급했다.
        2위 BYD와의 점유율 차이는 21.6%포인트, 3위 LG에너지솔루션과의 차이는 27.6%포인트에 이른다.
        한편, 국내 2차전지주는 이날 일제히 주가 하락을 맛봤다. LG에너지솔루션(-4.12%), 삼성SDI(-4.66%), SK이노베이션(-3.65%)과 에코프로(-6.58%) 등이 전날보다 하락 마감했다.
    """

    # 토큰 제한 테스트 함수
    async def test_token_truncation():
        # 긴 텍스트 생성 (반복)
        long_text = TEXT * 10  # 텍스트를 10번 반복하여 긴 텍스트 생성

        # 토큰 제한 테스트
        max_tokens = 100  # 테스트용 토큰 제한
        truncated = truncate_text_by_tokens(long_text, max_tokens)

        # 원본 및 잘린 텍스트의 토큰 수 계산
        encoding = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoding.encode(long_text))
        truncated_tokens = len(encoding.encode(truncated))

        print(f"원본 텍스트 토큰 수: {original_tokens}")
        print(f"잘린 텍스트 토큰 수: {truncated_tokens}")
        print(f"제한 토큰 수: {max_tokens}")
        print(f"잘린 텍스트 (처음 100자):\n{truncated[:100]}...")

        return truncated_tokens <= max_tokens

    async def single_task(index, host: Host, task: SystemRole):
        # 긴 텍스트 생성 (테스트용)
        long_text = TEXT * 5  # 텍스트를 5번 반복

        payload = {"text": long_text}
        try:
            start = time.perf_counter()
            result = await host.query(task, payload)
            end = time.perf_counter()
            print(
                f"[{index:03}] ✅ 응답 시간: {end - start:.2f}s / 결과 요약: {str(result['choices'][0]['message']['content'])[:40]}..."
            )
            ret.append(result["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"[{index:03}] ❌ 요청 실패: {e}")

    async def main():
        import dotenv, os

        dotenv.load_dotenv(override=True)
        # 토큰 제한 테스트 실행
        print("=== 토큰 제한 테스트 ===")
        success = await test_token_truncation()
        print(f"토큰 제한 테스트 결과: {'성공' if success else '실패'}")
        print("=" * 50)

        # 실제 API 호출 테스트
        print("\n=== API 호출 테스트 ===")
        async with Host(
            os.getenv("SERVER"),  # 실제 API 엔드포인트로 변경 필요
            os.getenv("MODEL"),  # 실제 모델 이름으로 변경 필요
            retry_attempts=2,  # 재시도 횟수 설정
            max_input_tokens=1000,  # 입력 토큰 제한 설정
            verbose=True,  # 상세 로깅 활성화
        ) as host:
            tasks = []
            sem = asyncio.Semaphore(CONCURRENCY)  # 동시 요청 수 제한

            async def limited_task(i):
                async with sem:
                    await single_task(i, host, SystemRole.TITLE)

            for i in range(NUM_REQUESTS):
                tasks.append(limited_task(i))

            # 실제 API 엔드포인트가 설정되어 있지 않으므로 주석 처리
            await asyncio.gather(*tasks)

    # 테스트 실행
    print("토큰 제한 기능이 추가된 Host 클래스 테스트")
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
