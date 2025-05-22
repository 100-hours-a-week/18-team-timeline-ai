# 18-team-timeline-ai
AI Repository

#### 실행 환경
- Linux, Cuda 12.4


#### 초기 설치
```bash
./scripts/init.sh 
```

#### poetry 설치 이후
```bash
poerty install
poetry run pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/
poerty install --with ** #(gpu, download)
```