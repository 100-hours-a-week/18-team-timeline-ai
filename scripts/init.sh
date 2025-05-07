#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Python 3.11 설치 확인 및 설치
if ! command -v python3.11 &> /dev/null; then
    log_info "Python 3.11이 설치되어 있지 않습니다. 설치를 시작합니다..."
    
    # macOS인 경우
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install python@3.11
    # Linux인 경우
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python3.11 python3.11-venv
    else
        log_error "지원하지 않는 운영체제입니다."
        exit 1
    fi
else
    log_info "Python 3.11이 이미 설치되어 있습니다."
fi

# Poetry 설치 확인 및 설치
if ! command -v poetry &> /dev/null; then
    log_info "Poetry가 설치되어 있지 않습니다. 설치를 시작합니다..."
    
    # macOS인 경우
    if [[ "$OSTYPE" == "darwin"* ]]; then
        curl -sSL https://install.python-poetry.org | python3 -
    # Linux인 경우
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sSL https://install.python-poetry.org | python3 -
    else
        log_error "지원하지 않는 운영체제입니다."
        exit 1
    fi
    
    # Poetry 환경 변수 설정
    export PATH="$HOME/.local/bin:$PATH"
else
    log_info "Poetry가 이미 설치되어 있습니다."
fi

# Poetry 버전 확인
poetry --version

# 프로젝트 의존성 파일 확인
if [ ! -f "pyproject.toml" ]; then
    log_error "pyproject.toml 파일이 없습니다. 프로젝트 루트 디렉토리에서 실행해주세요."
    exit 1
fi

# 필요한 패키지 설치
log_info "프로젝트 의존성을 설치합니다..."
poetry install --no-interaction

# 설치 확인
if [ $? -eq 0 ]; then
    log_info "의존성 설치가 완료되었습니다."
else
    log_error "의존성 설치에 실패했습니다."
    exit 1
fi

# Ollama 설치 확인
if ! command -v ollama &> /dev/null; then
    log_info "Ollama가 설치되어 있지 않습니다. 설치를 시작합니다..."
    
    # macOS인 경우
    if [[ "$OSTYPE" == "darwin"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    # Linux인 경우
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        log_error "지원하지 않는 운영체제입니다."
        exit 1
    fi
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
ollama pull bge-m3

# 다운로드 완료 확인
if [ $? -eq 0 ]; then
    log_info "bge-m3 모델 다운로드가 완료되었습니다."
else
    log_error "bge-m3 모델 다운로드에 실패했습니다."
    kill $OLLAMA_PID
    exit 1
fi

# Ollama 서비스 종료
log_info "Ollama 서비스를 종료합니다..."
kill $OLLAMA_PID

# HuggingFace 모델 다운로드
log_info "HuggingFace 모델을 다운로드합니다..."
poetry run python scripts/download_model.py

# 다운로드 완료 확인
if [ $? -eq 0 ]; then
    log_info "HuggingFace 모델 다운로드가 완료되었습니다."
else
    log_error "HuggingFace 모델 다운로드에 실패했습니다."
    exit 1
fi

log_info "초기화가 완료되었습니다."
