#!/bin/bash

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="${SCRIPT_DIR}/init.log"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# 초기화 함수
init() {
    # 로그 파일 초기화
    echo "=== 초기화 시작: $(date) ===" > "$LOG_FILE"
    
    # 종료 시 정리 작업을 위한 트랩 설정
    trap cleanup EXIT
    
    # 필요한 디렉토리 생성
    mkdir -p "${SCRIPT_DIR}/models"
    mkdir -p "${SCRIPT_DIR}/datasets"
    mkdir -p "${SCRIPT_DIR}/logs"
    
    # Python 3.11 설치 확인 및 설치
    install_python
    
    # Poetry 설치 확인 및 설치
    install_poetry
    
    # 프로젝트 의존성 설치
    install_dependencies
    
    # Ollama 설치 및 모델 다운로드
    install_ollama
    
    # 데이터셋 다운로드
    download_dataset
    
    # HuggingFace 모델 다운로드
    download_model
    
    log_info "초기화가 완료되었습니다."
}

# 종료 시 정리 작업
cleanup() {
    # Ollama 서비스가 실행 중이면 종료
    if [ -n "$OLLAMA_PID" ] && ps -p $OLLAMA_PID > /dev/null; then
        log_info "Ollama 서비스를 종료합니다..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    
    echo "=== 초기화 종료: $(date) ===" >> "$LOG_FILE"
}

# Python 3.11 설치
install_python() {
    if ! command -v python3.11 &> /dev/null; then
        log_info "Python 3.11이 설치되어 있지 않습니다. 설치를 시작합니다..."
        
        # macOS인 경우
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install python@3.11 || {
                log_error "Python 3.11 설치 실패"
                exit 1
            }
        # Linux인 경우
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt update || {
                log_error "apt update 실패"
                exit 1
            }
            sudo apt install -y software-properties-common || {
                log_error "software-properties-common 설치 실패"
                exit 1
            }
            sudo add-apt-repository -y ppa:deadsnakes/ppa || {
                log_error "PPA 추가 실패"
                exit 1
            }
            sudo apt update || {
                log_error "apt update 실패"
                exit 1
            }
            sudo apt install -y python3.11 python3.11-venv python3.11-dev || {
                log_error "Python 3.11 설치 실패"
                exit 1
            }
        else
            log_error "지원하지 않는 운영체제입니다."
            exit 1
        fi
        
        log_info "Python 3.11 설치 완료"
    else
        log_info "Python 3.11이 이미 설치되어 있습니다."
    fi
    
    # Python 버전 확인
    python3.11 --version
}

# Poetry 설치
install_poetry() {
    if ! command -v poetry &> /dev/null; then
        log_info "Poetry가 설치되어 있지 않습니다. 설치를 시작합니다..."
        
        # Poetry 설치
        curl -sSL https://install.python-poetry.org | python3 - || {
            log_error "Poetry 설치 실패"
            exit 1
        }
        
        # Poetry 환경 변수 설정
        export PATH="$HOME/.local/bin:$PATH"
        
        # .bashrc에 PATH 추가 (영구 설정)
        if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            log_info "Poetry PATH를 .bashrc에 추가했습니다."
        fi
        
        log_info "Poetry 설치 완료"
    else
        log_info "Poetry가 이미 설치되어 있습니다."
    fi
    
    # Poetry 버전 확인
    poetry --version
}

# 프로젝트 의존성 설치
install_dependencies() {
    # 프로젝트 루트 디렉토리로 이동
    cd "$SCRIPT_DIR" || {
        log_error "프로젝트 디렉토리로 이동 실패"
        exit 1
    }
    
    # pyproject.toml 파일 확인
    if [ ! -f "pyproject.toml" ]; then
        log_warn "pyproject.toml 파일이 없습니다. 기본 파일을 생성합니다."
        
        # 기본 pyproject.toml 파일 생성
        cat > pyproject.toml << EOF
[project]
name = "dev"
version = "0.1.0"
description = ""
authors = [
    {name = "phur99",email = "skerrigen12@gmail.com"}
]
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "python-dotenv (>=1.1.0,<2.0.0)",
    "newspaper3k (>=0.2.8,<0.3.0)",
    "lxml-html-clean (>=0.4.2,<0.5.0)",
    "pykakao (>=0.0.7,<0.0.8)",
    "google-auth-oauthlib (>=1.2.1,<2.0.0)",
    "google-auth-httplib2 (>=0.2.0,<1.0.0)",
    "google-api-python-client (>=2.166.0,<3.0.0)",
    "slowapi (==0.1.9)",
    "tqdm (>=4.67.1,<5.0.0)",
    "aiolimiter (>=1.2.1,<2.0.0)",
    "fastapi (==0.115.8)",
    "pydantic (==2.10.6)",
    "qdrant-client[grpc] (>=1.14.2,<2.0.0)",
    "orjson (>=3.10.18,<4.0.0)",
    "uvicorn (>=0.34.2,<0.35.0)",

]

[[tool.poetry.source]]
name = "flashinfer"
url = "https://flashinfer.ai/whl/cu124/torch2.6/"
priority = "explicit"

[tool.poetry.group.gpu.dependencies]
vllm = "==0.8.5"
pyzmq = "==25.1.1"

[tool.poetry.group.test.dependencies]
pytest = ">=8.3.5,<9.0.0"
pytest-asyncio = ">=0.26.0,<0.27.0"
pytest-tornasync = ">=0.6.0.post2,<0.7.0"
pytest-trio = ">=0.8.0,<0.9.0"
pytest-twisted = ">=1.14.3,<2.0.0"
line-profiler = ">=4.2.0,<5.0.0"
anyio = ">=4.9.0,<5.0.0"
twisted = ">=24.11.0,<25.0.0"

[tool.poetry.group.download.dependencies]
huggingface-hub = ">=0.30.0,<1.0.0"
datasets  = ">=3.6.0,<4.0.0"


[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

EOF
        log_info "기본 pyproject.toml 파일이 생성되었습니다."
    fi
    
    # 필요한 패키지 설치
    log_info "프로젝트 의존성을 설치합니다..."
    poetry install --no-interaction || {
        log_error "의존성 설치 실패"
        exit 1
    }
    
    log_info "의존성 설치가 완료되었습니다."
}

# Ollama 설치 및 모델 다운로드
install_ollama() {
    # Ollama 설치 확인
    if ! command -v ollama &> /dev/null; then
        log_info "Ollama가 설치되어 있지 않습니다. 설치를 시작합니다..."
        
        # macOS인 경우
        if [[ "$OSTYPE" == "darwin"* ]]; then
            curl -fsSL https://ollama.com/install.sh | sh || {
                log_error "Ollama 설치 실패"
                exit 1
            }
        # Linux인 경우
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -fsSL https://ollama.com/install.sh | sh || {
                log_error "Ollama 설치 실패"
                exit 1
            }
        else
            log_error "지원하지 않는 운영체제입니다."
            exit 1
        fi
        
        log_info "Ollama 설치 완료"
    else
        log_info "Ollama가 이미 설치되어 있습니다."
    fi
    
    # Ollama 서비스 시작
    log_info "Ollama 서비스를 시작합니다..."
    ollama serve &
    OLLAMA_PID=$!
    
    # 서비스가 시작될 때까지 대기
    sleep 5
    
    # bge-m3 모델 다운로드
    log_info "bge-m3 모델을 다운로드합니다..."
    ollama pull bge-m3 || {
        log_error "bge-m3 모델 다운로드 실패"
        kill $OLLAMA_PID
        exit 1
    }
    
    log_info "bge-m3 모델 다운로드가 완료되었습니다."
    
    # Ollama 서비스 종료
    log_info "Ollama 서비스를 종료합니다..."
    kill $OLLAMA_PID
    unset OLLAMA_PID
}

# 데이터셋 다운로드
download_dataset() {
    log_info "데이터셋을 다운로드합니다..."
    
    # 설정 파일 확인 및 생성
    if [ ! -d "${SCRIPT_DIR}/config" ]; then
        mkdir -p "${SCRIPT_DIR}/config"
    fi
    
    # settings.py 파일이 없으면 생성
    if [ ! -f "${SCRIPT_DIR}/config/settings.py" ]; then
        cat > "${SCRIPT_DIR}/config/settings.py" << EOF
# 데이터셋 설정
DATASET_NAME = "squad"
DATASET_CACHE_DIR = "${SCRIPT_DIR}/datasets"
EOF
        log_info "config/settings.py 파일이 생성되었습니다."
    fi
    
    # __init__.py 파일 생성
    touch "${SCRIPT_DIR}/config/__init__.py"
    
    # 데이터셋 다운로드 스크립트 실행
    cd "$SCRIPT_DIR" || {
        log_error "프로젝트 디렉토리로 이동 실패"
        exit 1
    }
    
    poetry run python "${SCRIPT_DIR}/download_dataset.py" || {
        log_error "데이터셋 다운로드 실패"
        exit 1
    }
    
    log_info "데이터셋 다운로드가 완료되었습니다."
}

# HuggingFace 모델 다운로드
download_model() {
    log_info "HuggingFace 모델을 다운로드합니다..."
    
    # .env 파일 확인 및 생성
    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        log_warn ".env 파일이 없습니다. 기본 파일을 생성합니다."
        
        # 사용자에게 토큰 입력 요청
        read -p "HuggingFace 토큰을 입력하세요 (없으면 Enter): " hf_token
        
        # .env 파일 생성
        echo "HF_TOKEN=${hf_token}" > "${SCRIPT_DIR}/.env"
        log_info ".env 파일이 생성되었습니다."
    fi
    
    # 모델 다운로드 스크립트 실행
    cd "$SCRIPT_DIR" || {
        log_error "프로젝트 디렉토리로 이동 실패"
        exit 1
    }
    
    poetry run python "${SCRIPT_DIR}/download_model.py" || {
        log_error "HuggingFace 모델 다운로드 실패"
        exit 1
    }
    
    log_info "HuggingFace 모델 다운로드가 완료되었습니다."

}

# 메인 함수 실행
init
