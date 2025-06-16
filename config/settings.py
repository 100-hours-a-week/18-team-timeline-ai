# ------------------------------------------------------------------------------
# classify 설정
# Qdrant 설정
import os
from dotenv import load_dotenv
from utils.logger import Logger

load_dotenv(override=True)

logger = Logger.get_logger("settings")

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not all([QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY]):
    logger.error("Qdrant 환경변수가 설정되지 않았습니다.")

VECTOR_SIZE = 1024
BATCH_SIZE = 32
# 감정 레이블 설정
LABELS = [
    "불평/불만",
    "환영/호의",
    "감동/감탄",
    "지긋지긋",
    "고마움",
    "슬픔",
    "화남/분노",
    "존경",
    "기대감",
    "우쭐댐/무시함",
    "안타까움/실망",
    "비장함",
    "의심/불신",
    "뿌듯함",
    "편안/쾌적",
    "신기함/관심",
    "아껴주는",
    "부끄러움",
    "공포/무서움",
    "절망",
    "한심함",
    "역겨움/징그러움",
    "짜증",
    "어이없음",
    "없음",
    "패배/자기혐오",
    "귀찮음",
    "힘듦/지침",
    "즐거움/신남",
    "깨달음",
    "죄책감",
    "증오/혐오",
    "흐뭇함(귀여움/예쁨)",
    "당황/난처",
    "경악",
    "부담/안_내킴",
    "서러움",
    "재미없음",
    "불쌍함/연민",
    "놀람",
    "행복",
    "불안/걱정",
    "기쁨",
    "안심/신뢰",
]
DICT_LABELS = {i: label for i, label in enumerate(LABELS)}

SENTIMENT_MAP = {
    # 긍정
    "환영/호의": "긍정",
    "감동/감탄": "긍정",
    "고마움": "긍정",
    "존경": "긍정",
    "기대감": "긍정",
    "뿌듯함": "긍정",
    "편안/쾌적": "긍정",
    "신기함/관심": "긍정",
    "즐거움/신남": "긍정",
    "깨달음": "긍정",
    "흐뭇함(귀여움/예쁨)": "긍정",
    "행복": "긍정",
    "기쁨": "긍정",
    "안심/신뢰": "긍정",
    "아껴주는": "긍정",
    # 부정
    "불평/불만": "부정",
    "지긋지긋": "부정",
    "슬픔": "부정",
    "화남/분노": "부정",
    "우쭐댐/무시함": "부정",
    "안타까움/실망": "부정",
    "의심/불신": "부정",
    "부끄러움": "부정",
    "공포/무서움": "부정",
    "절망": "부정",
    "한심함": "부정",
    "역겨움/징그러움": "부정",
    "짜증": "부정",
    "어이없음": "부정",
    "패배/자기혐오": "부정",
    "귀찮음": "부정",
    "힘듦/지침": "부정",
    "죄책감": "부정",
    "증오/혐오": "부정",
    "당황/난처": "부정",
    "경악": "부정",
    "부담/안_내킴": "부정",
    "서러움": "부정",
    "재미없음": "부정",
    "불안/걱정": "부정",
    # 중립
    "없음": "중립",
    "불쌍함/연민": "중립",
    "놀람": "중립",
    "비장함": "중립",
}
# ------------------------------------------------------------------------------
# 데이터셋 설정
DATASET_NAME = "searle-j/kote"
DATASET_CACHE_DIR = ".dataset"
COLLECTION_NAME = "kote"
DATASET_VOLUME = "./qdrant_storage"
# ------------------------------------------------------------------------------
# OLLAMA 설정
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
OLLAMA_MODEL = "bge-m3"

if not all([OLLAMA_HOST, OLLAMA_PORT]):
    logger.error("Ollama 환경변수가 설정되지 않았습니다.")

# ------------------------------------------------------------------------------
# 밴 방지용 설정
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36"
POOL_CONECTION = 100
POOL_MAXSIZE = 100
MAX_RETRIES = 3
ARTICLE_TIMEOUT = 10
