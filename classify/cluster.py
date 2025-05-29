import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class NewsClassifier:
    def __init__(self, threshold=0.2):
        self.threshold = threshold  # 기타 판단 기준
        self.model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device="cpu")

        # 대표 뉴스: 각 카테고리당 여러 문장 사용
        self.category_examples = {
            "경제": [
                "분양가 10년새 2배 껑충, 제로 에너지 인증에 추가 상승 전망",
                "SK하이닉스 좋아요, 삼성전자 글쎄, 외국인 돌아온 국내증시",
                "결혼비용 평균 2천101만원, 강남은 경상도의 세 배 달해",
                "산업부, 올해 산업AI 개발·확산에 4800억원 투자한다",
                "트럼프발 무역전쟁, 최악은 지났다. 미-EU 긴장 완화에 증시 급등"
            ],
            "스포츠": [
                "무관의 한 털어낸 손흥민… UEL 메달 퍼포먼스로 귀국 인사",
                "수원KT소닉붐 프로농구단, KCC 정창영 영입",
                "김연아가 아이스쇼에서 완벽한 연기를 펼쳤다",
                "추락하는 키움… 프로야구 최초 시즌 100패 우려",
                "LCK 9주차 돌입, T1·한화생명, 2위 놓고 격돌"
            ],
            "연예": [
                "BTS가 빌보드 차트 1위를 차지했다",
                "고민시, 학교폭력 논란에 시청률 흔들",
                "이동건, 미모의 여성과 열애설, 소속사 대응은",
                "갓세븐 박진영, 3개월간 혹독 다이어트",
                "120억 소송 가세연, 김수현 폭로 역풍 맞았다"
            ],
            "과학": [
                "구글·삼성 XR 무한 프로젝트 출시 임박",
                "세종대 채규현 교수, 장주기 쌍성의 3차원 속도에 의한 중력 측정",
                "AI가 스스로 코드 조작, 인간 명령 거부한 첫 사례 나왔다",
                "손가락 끝 감각 되살릴 로봇손, 내년 첫 임상시험",
                "뇌심부자극기 이식 환자에 고집적 초음파 수술, 세계 첫 성공"
            ]
        }

        # 대표 문장들을 미리 임베딩하여 평균 벡터 저장
        self.category_vectors = self._precompute_category_vectors()

    def _precompute_category_vectors(self):
        # 카테고리별 대표 문장 평균 벡터 계산
        category_vectors = {}
        for category, examples in self.category_examples.items():
            vecs = self.model.encode(examples, convert_to_tensor=True).cpu().numpy()
            avg_vec = np.mean(vecs, axis=0)
            category_vectors[category] = avg_vec
        return category_vectors

    def classify(self, news_text: str) -> str:
        news_vec = self.model.encode([news_text], convert_to_tensor=True).cpu().numpy()

        similarities = {
            category: cosine_similarity(news_vec, avg_vec.reshape(1, -1))[0, 0]
            for category, avg_vec in self.category_vectors.items()
        }

        max_category = max(similarities, key=similarities.get)
        max_score = similarities[max_category]

        if max_score < self.threshold:
            return "기타"
        else:
            return max_category
