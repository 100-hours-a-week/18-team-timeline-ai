import os
from datasets import load_dataset
from config.settings import DATASET_NAME, DATASET_CACHE_DIR

# .dataset 폴더 생성
dataset_dir = DATASET_CACHE_DIR
os.makedirs(dataset_dir, exist_ok=True)

# 데이터셋 다운로드 및 저장
ds = load_dataset(
    DATASET_NAME,
    cache_dir=dataset_dir,
    trust_remote_code=True,  # 원격 코드 실행 허용
)

print(f"데이터셋이 {dataset_dir} 폴더에 저장되었습니다.")
print(f"데이터셋 크기: {len(ds['train'])} 학습 샘플, {len(ds['test'])} 테스트 샘플")
