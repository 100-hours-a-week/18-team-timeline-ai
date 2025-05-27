# 18-team-timeline-ai
AI Repository

#### 실행 환경
- MacOS, Cuda 12.4


#### 초기 설치(init.sh 필요)
```bash
chmod +x init.sh
./init.sh 
```

#### poetry 설치 이후
```bash
poerty install
poetry run pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/
```

### vLLM 서버
```bash
poerty install --only gpu
```

### 모델 다운로드
```bash
poetry install --with download
```

### 모델 다운로드 받은 이후(모델 다운로드 패키지 제거)
```bash
poetry remove datasets huggingface-hub
```