import asyncio
from typing import List, Dict
from collections import defaultdict
from ai_models.graph.host import Host, SystemRole
from utils.logger import Logger
import logging
import time

logger = Logger.get_logger("ai_models.manager", log_level=logging.ERROR)


class BatchManager:
    def __init__(
        self,
        host: Host,
        batch_size: int = 4,
        max_wait_time: float = 1.0,
    ):
        self.host = host
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = asyncio.Queue()
        self.running = False

    async def submit(self, task: SystemRole, payload: dict):
        await self.queue.put((task, payload))

    async def _gather_batch(self) -> List[dict]:
        batch = []
        start = asyncio.get_event_loop().time()
        while len(batch) < self.batch_size:
            timeout = self.max_wait_time - (asyncio.get_event_loop().time() - start)
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch

    async def run(self):
        self.running = True
        while self.running:
            batch = await self._gather_batch()
            if not batch:
                continue
            await self.process_batch(batch)

    async def process_batch(self, batch: List[dict]):
        task_map: Dict[SystemRole, List[dict]] = defaultdict(list)
        for task_type, payload in batch:
            task_map[task_type].append(payload)

        for task_type, payloads in task_map.items():
            for payload in payloads:
                try:
                    result = await self.host.query(task_type, payload)
                    logger.info(
                        f"[{task_type.name} ✅] {result['choices'][0]['message']['content']}"
                    )
                except Exception as e:
                    logger.error(f"[{task_type.name} ❌] 요청 실패: {e}")


async def main():
    async with Host(
        "http://b5c5-34-118-242-65.ngrok-free.app",
        "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    ) as host:
        manager = BatchManager(host, batch_size=6, max_wait_time=0.5)
        runner = asyncio.create_task(manager.run())
        TEXT = """
            글로벌 배터리 시장의 38%를 점유하는 ‘대어’ CATL이 홍콩 주식시장에 신규 상장했다. 상장 첫날 CATL 주가는 공모가보다 최대 18% 높은 가격에 거래됐다.
            국내 개인투자자들은 이번 기업공개(IPO)로 인해 CATL을 직접투자할 수 있는 길이 열렸다.
            20일(현지시간) 홍콩증권거래소에서 CATL은 장 초반 공모가인 263홍콩달러(약 4만6800원)보다 약 12.5% 높은 296홍콩달러(약 5만2700원)에 거래됐다.
            CATL은 이날 정오께 311.4달러(약 5만5400원)에 거래되며 최고가를 기록했다. 시초가 대비로는 5.2%, 공모가 대비로는 18.4% 높은 가격이다.
            오후 3시께 CATL은 307.6홍콩달러(약 5만4700원)에 거래됐다.
            CATL은 이번 IPO를 통해 46억달러(약 6조4000억원) 이상을 조달한 것으로 전해진다. 초과 배정 옵션을 행사할 경우 총 조달액은 53억달러(약 7조3000억원)까지 불어날 수 있다.
            이는 올해 전 세계 IPO 시장에서 최대 규모 금액이다. 지난해 홍콩증시에 상장했던 중국의 가전업체 메이디(46억달러)의 사례도 뛰어넘는다.
            2021년에 62억달러를 조달했던 중국의 온라인 플랫폼 기업 콰이쇼우테크놀로지와도 비견된다.
            CATL은 조달 금액의 90% 이상을 헝가리 공장 건설에 사용할 계획이다. 2027년까지 완공 예정인 이번 프로젝트를 통해 CATL은 유럽시장을 더욱 확장할 전망이다.
            이번 IPO 과정에선 중국석유화공(시노펙)과 쿠웨이트투자청, 카타르투자청, 힐하우스인베스트먼트, 오크트리캐피털 등이 주요 투자자로 참여했다.
            공모청약의 1억2540만주는 기관 투자자에게, 1016만주는 홍콩 개인 투자자에게 매각됐다. 이 과정에서 미국 개인투자자의 공모 참여를 제한하는 ‘레귤레이션 S’ 방식이 활용되기도 했다.
            IPO 주관사는 중국국제금융공사(CICC)와 더불어 뱅크오브아메리카, 골드만삭스, 모건스탠리, JP모건등이 맡았다.
            CATL이 홍콩증시에 입성하면서 국내 개인투자자들에게도 CATL 직접투자의 길이 열렸다.
            CATL은 지난 2018년 중국 본토 선전증권거래소에 상장했지만, 이는 선전증시와 홍콩증시를 잇는 ‘선강퉁 제도’에 포함돼지 않아 외국인 개인투자자들의 매수가 사실상 불가능했다.
            이에 국내 개인투자자는 CATL이 포함된 상장지수펀드(ETF)를 매수하는 것이 최선이었다. 그러나 이번 상장을 통해 직접 매매가 가능해졌다.
            존슨 완 제프리스 중국 연구원은 이날 “CATL은 견조한 실적과 매력적 밸류에이션이 있어 앞으로 50% 이상 상승할 수 있다”며 CATL의 주가 성장 가능성을 높게 점쳤다.
            중국의 ‘배터리 굴기’를 대표하는 CATL은 이미 글로벌 배터리 산업을 주도하고 있다.
            SNE리서치에 따르면 CATL은 올해 1분기 기준으로 글로벌 배터리시장의 38.3%에 해당하는 84.9기가와트시(GWh)를 공급했다.
            2위 BYD와의 점유율 차이는 21.6%포인트, 3위 LG에너지솔루션과의 차이는 27.6%포인트에 이른다.
            한편, 국내 2차전지주는 이날 일제히 주가 하락을 맛봤다. LG에너지솔루션(-4.12%), 삼성SDI(-4.66%), SK이노베이션(-3.65%)과 에코프로(-6.58%) 등이 전날보다 하락 마감했다.
        """
        start = time.perf_counter()
        for i in range(10):
            for j in range(3):
                await manager.submit(SystemRole.SUMMARIZE, {"text": TEXT})
                await manager.submit(SystemRole.TITLE, {"text": TEXT})
                await manager.submit(SystemRole.TAG, {"text": TEXT})
        end = time.perf_counter()
        print(f"총 실행 시간: {end - start:.2f}s")
        await asyncio.sleep(3)
        manager.running = False
        await runner


if __name__ == "__main__":
    asyncio.run(main())
