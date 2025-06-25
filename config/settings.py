# ------------------------------------------------------------------------------
# classify 설정
# Qdrant 설정 (services/classify.py, utils/storage.py 등)
import os
from dotenv import load_dotenv
from utils.logger import Logger
from utils.env_check import check_env_vars

load_dotenv(override=True)

logger = Logger.get_logger("settings")

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

QDRANT_VARS = {
    "QDRANT_HOST": QDRANT_HOST,
    "QDRANT_PORT": QDRANT_PORT,
    "QDRANT_API_KEY": QDRANT_API_KEY,
}
check_env_vars(QDRANT_VARS, logger, prefix="Qdrant: ")

VECTOR_SIZE = 1024
BATCH_SIZE = 32
# 감정 레이블 설정 (services/classify.py 등)
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
# 데이터셋 설정 (scripts/make_db.py 등)
COMMENT_DATASET_NAME = "searle-j/kote"
COMMENT_DATASET_CACHE_DIR = ".dataset"
COMMENT_COLLECTION_NAME = "kote"
COMMENT_DATASET_VOLUME = "./qdrant_storage"
# ------------------------------------------------------------------------------
# OLLAMA 설정 (inference/embedding.py, scripts/make_db.py, api/comment.py, api/timeline.py)
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
if not OLLAMA_HOST:
    logger.error("OLLAMA_HOST 환경변수가 설정되지 않았습니다.")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
if not OLLAMA_PORT:
    logger.error("OLLAMA_PORT 환경변수가 설정되지 않았습니다.")
OLLAMA_MODELS = ["bge-m3", "bge-m3"]

# ------------------------------------------------------------------------------
# 밴 방지용 설정
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/113.0.0.0 Safari/537.36"
POOL_CONECTION = 100
POOL_MAXSIZE = 100
MAX_RETRIES = 3
ARTICLE_TIMEOUT = 10
# ------------------------------------------------------------------------------
# tagger.py
TAG_LABELS = {
    1: "경제",
    2: "연예",
    3: "스포츠",
}

THRESHOLD = 0.2  # 코사인 유사도 기준 미만이면 기타로 분류
TAG_COLLECTION_NAME = "tag"
# ------------------------------------------------------------------------------
# timeline.py
TAG_NAMES = ["", "ECONOMY", "ENTERTAINMENT", "SPORTS", "SCIENCE"]
BASE_IMG_URL = "https://github.com/user-attachments/assets/"
IMG_LINKS = [
    "1eeef1f6-3e0a-416a-bc4d-4922b27db855",
    "6cf88794-2743-4dd1-858c-4fcd76f8f107",
    "35ee8d58-b5d8-47c0-82e8-38073f4193eb",
    "3f4248cb-7d8d-4532-a71a-2346e8a82957",
    "e3b550d9-1d62-4940-b942-5b431ba6674e",
]

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    logger.error("YOUTUBE_API_KEY 환경변수가 설정되지 않았습니다.")
REST_API_KEY = os.getenv("REST_API_KEY")
if not REST_API_KEY:
    logger.error("REST_API_KEY 환경변수가 설정되지 않았습니다.")
MAX_COMMENTS = 10

MAX_WORKERS = 6
# ------------------------------------------------------------------------------
# article_extractor
DEFAULT_LANG = "ko"
DOMAIN_TIMEOUTS = {
    "sportivomedia.net": 30,
    "default": ARTICLE_TIMEOUT,
}
DOMAIN_RETRIES = {
    "sportivomedia.net": 5,
    "default": 3,
}
CLIENT_TIMEOUT = 30
# ------------------------------------------------------------------------------
# timeline.py에서 사용
SERVER = os.getenv("SERVER")
if not SERVER:
    logger.error("SERVER 환경변수가 설정되지 않았습니다.")
MODEL = os.getenv("MODEL")
if not MODEL:
    logger.error("MODEL 환경변수가 설정되지 않았습니다.")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

# ------------------------------------------------------------------------------
# SERP/Serper API 키 (api/hot.py 등)
SERP_API_KEYS = os.getenv("SERP_API_KEYS", "").split(",")
if not SERP_API_KEYS or SERP_API_KEYS == [""]:
    logger.error("SERP_API_KEYS 환경변수가 설정되지 않았습니다.")
SERPER_API_KEYS = os.getenv("SERPER_API_KEYS", "").split(",")
if not SERPER_API_KEYS or SERPER_API_KEYS == [""]:
    logger.error("SERPER_API_KEYS 환경변수가 설정되지 않았습니다.")


def get_serp_key(i: int):
    if not SERP_API_KEYS or i < 0 or i >= len(SERP_API_KEYS):
        return ""
    return SERP_API_KEYS[i].strip()


def get_serper_key(i: int):
    if not SERPER_API_KEYS or i < 0:
        return ""
    if i >= len(SERPER_API_KEYS):
        return ""
    import random

    return random.choice(SERPER_API_KEYS).strip()


# ------------------------------------------------------------------------------
# Gemini API 키 (scrapers/serper.py 등)
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "").split(",")
if not GEMINI_API_KEYS or GEMINI_API_KEYS == [""]:
    logger.error("GEMINI_API_KEYS 환경변수가 설정되지 않았습니다.")
