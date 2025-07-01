# 18-team-timeline-ai

AI 기반 타임라인 분석 및 생성 시스템

## 프로젝트 개요

이 프로젝트는 다양한 소스에서 데이터를 수집하고, AI를 활용하여 타임라인을 생성하고 분석하는 시스템입니다.

### 주요 기능
- 다양한 소스(웹, API 등)에서 데이터 수집
- AI 기반 타임라인 생성 및 분석
- 실시간 데이터 처리 및 업데이트
- RESTful API를 통한 데이터 접근

## 기술 스택

### 백엔드
- FastAPI
- Python 3.11+
- Poetry (의존성 관리)
- Qdrant (벡터 데이터베이스)
- Sentence Transformers
- OpenAI API

### 인프라
- Docker
- CUDA 12.4 (GPU 지원)

## 설치 및 실행

### 필수 요구사항
- Linux 운영체제
- CUDA 12.4 지원 GPU
- Python 3.11 이상
- Poetry

### 초기 설치
```bash
# 초기 설정 스크립트 실행
./scripts/init.sh 

# Poetry 의존성 설치
poetry install

# GPU 지원 패키지 설치
poetry run pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/

# 추가 의존성 설치 (GPU, 다운로드 등)
poetry install --with gpu,download
```

### 환경 설정
1. `.env` 파일을 프로젝트 루트에 생성
2. 필요한 환경 변수 설정:
   ```
   OPENAI_API_KEY=your_api_key
   QDRANT_URL=your_qdrant_url
   ```

## 프로젝트 구조
```
.
├── api/                    # API 엔드포인트 정의
│   ├── router.py          # 메인 라우터 설정
│   ├── timeline.py       # 타임라인 관련 API
│   ├── hot.py            # 인기 콘텐츠 관련 API
│   ├── comment.py        # 댓글 관련 API
│   └── merge.py          # 데이터 병합 관련 API
│
├── config/                # 설정 파일
│   ├── settings.py       # 기본 설정
│   └── constants.py      # 상수 정의
│
├── docker/                  # Docker 관련 파일
│   ├── Dockerfile          # 메인 Dockerfile
│   └── docker-compose.yml  # 서비스 구성
│
├── inference/           # AI 추론 관련 코드
│   ├── models/         # 추론 모델
│   └── pipelines/      # 추론 파이프라인
│
├── models/             # 데이터 모델
│   ├── timeline.py    # 타임라인 모델
│   └── user.py        # 사용자 모델
│
├── pipelines/          # 데이터 처리 파이프라인
│   ├── collector/     # 데이터 수집
│   ├── processor/     # 데이터 처리
│   └── analyzer/      # 데이터 분석
│
├── scrapers/          # 데이터 수집기
│   ├── web/          # 웹 스크래핑
│   └── api/          # API 데이터 수집
│
├── services/           # 비즈니스 로직
│   ├── news_tag.py    # 뉴스 분류 서비스
│   └── sentiment.py   # 댓글 분류 서비스
│
├── utils/            # 유틸리티 함수
│   ├── logger.py    # 로깅 유틸리티
│   ├── cache.py     # 캐싱 유틸리티
│   └── helpers.py   # 헬퍼 함수
│
├── test/               # 테스트 코드
│   ├── unit/          # 단위 테스트
│   └── integration/   # 통합 테스트
│
├── schemas/         # 데이터 스키마
│   ├── request.py  # 요청 스키마
│   └── response.py # 응답 스키마
│
├── logs/           # 로그 파일
├── .dataset/       # 데이터셋 저장소
├── qdrant_storage/ # 벡터 데이터베이스 저장소
│
├── main.py        # 애플리케이션 진입점
├── pyproject.toml # 프로젝트 설정
└── poetry.lock    # 의존성 잠금 파일
```

각 디렉토리의 주요 역할:

### API (`/api`)
- RESTful API 엔드포인트 정의
- 요청/응답 처리
- 라우팅 설정

### 설정 (`/config`)
- 환경 설정
- 상수 정의
- 설정 파일 관리

### Docker (`/docker`)
- 컨테이너화 설정
- 서비스 구성
- 배포 설정

### 추론 (`/inference`)
- AI 모델 추론
- 추론 파이프라인
- 모델 관리

### 모델 (`/models`)
- 데이터 모델 정의
- ORM 모델
- 데이터 구조

### 파이프라인 (`/pipelines`)
- 데이터 수집 파이프라인
- 데이터 처리 파이프라인
- 분석 파이프라인

### 스크래퍼 (`/scrapers`)
- 웹 스크래핑
- API 데이터 수집
- 데이터 수집 유틸리티

### 서비스 (`/services`)
- 비즈니스 로직
- 핵심 기능 구현
- 서비스 레이어

### 유틸리티 (`/utils`)
- 공통 유틸리티 함수
- 로깅
- 캐싱
- 헬퍼 함수

### 테스트 (`/test`)
- 단위 테스트
- 통합 테스트
- 테스트 유틸리티

### 스키마 (`/schemas`)
- 요청/응답 스키마
- 데이터 검증
- API 문서화

## API 문서

API 문서는 서버 실행 후 다음 URL에서 확인할 수 있습니다:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 개발 가이드

### 코드 스타일
- PEP 8 가이드라인 준수
- 타입 힌트 사용
- 문서화 주석 필수

### 테스트
```bash
# 테스트 실행
poetry run pytest

# 특정 테스트 실행
poetry run pytest test/test_specific.py
```

## 모니터링

OpenTelemetry를 사용하여 다음 메트릭을 모니터링합니다:
- API 응답 시간
- 에러율
- 리소스 사용량
- 시스템 상태

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여 방법

1. 이슈 생성
2. 브랜치 생성
3. 변경사항 커밋
4. Pull Request 생성

## 연락처

프로젝트 관리자: phur99, Lockway
