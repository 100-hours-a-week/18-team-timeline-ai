import os
import logging
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """환경 변수 설정 및 HuggingFace 로그인"""
    # .env 파일 로드
    load_dotenv()

    # HuggingFace 토큰 확인
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN이 .env 파일에 설정되지 않았습니다.")

    # HuggingFace 로그인
    try:
        login(token=hf_token)
        logger.info("HuggingFace 로그인 성공")
    except Exception as e:
        logger.error(f"HuggingFace 로그인 실패: {str(e)}")
        raise


def download_model(model_id: str, cache_dir: str = "./models"):
    """모델 다운로드 및 저장

    Args:
        model_id (str): HuggingFace 모델 ID
        cache_dir (str): 모델 저장 디렉토리
    """
    try:
        # 저장 디렉토리 생성
        os.makedirs(cache_dir, exist_ok=True)

        # 모델 다운로드
        logger.info(f"모델 다운로드 시작: {model_id}")
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, model_id.split("/")[-1]),
            local_dir_use_symlinks=False,
        )
        logger.info(f"모델 다운로드 완료: {model_id}")

        # 토크나이저와 모델 로드 테스트
        logger.info("모델 로드 테스트 중...")
        model_path = "models/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

        logger.info(f"모델이 {model_path}에 저장되었습니다")

    except Exception as e:
        logger.error(f"모델 다운로드 실패: {str(e)}")
        raise


def main():
    """메인 함수"""
    # 다운로드할 모델 목록
    models = ["naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"]

    try:
        # 환경 설정 및 로그인
        setup_environment()

        # 각 모델 다운로드
        for model_id in models:
            download_model(model_id)

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    main()
