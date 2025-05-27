# 18-team-timeline-ai
AI Repository

#### 실행 환경
- MacOS/Ubuntu, Cuda 12.4(GPU)


#### 초기 설치(init.sh 필요)
```bash
chmod +x init.sh #접근 권한 없을 시에
./init.sh #main만 설치
./init.sh main gpu #main, gpu 설치
./init.sh main download 
./init.sh gpu 
```

#### init.sh 설치 이후(GPU 서버)
```bash
poetry run pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/
```

### 모델 다운로드 받은 이후(모델 다운로드 패키지 제거)
```bash
poetry remove datasets huggingface-hub torch transformers
```