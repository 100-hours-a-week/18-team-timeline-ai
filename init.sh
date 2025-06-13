#!/bin/bash

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LOG_FILE="${SCRIPT_DIR}/init.log"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}
log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}
log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

init() {
    echo "=== 초기화 시작: $(date) ===" > "$LOG_FILE"
    trap cleanup EXIT

    mkdir -p "${SCRIPT_DIR}/logs"
    install_homebrew
    install_python
    setup_python_path
    install_poetry
    install_dependencies "$@"
    install_ollama
    install_qdrant
    download_model
    download_dataset
    log_info "초기화가 완료되었습니다."
}

cleanup() {
    if [ -n "$OLLAMA_PID" ] && ps -p $OLLAMA_PID > /dev/null; then
        log_info "Ollama 서비스를 종료합니다..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    echo "=== 초기화 종료: $(date) ===" >> "$LOG_FILE"
}
install_homebrew() {
    if command -v brew &>/dev/null; then
        log_info "Homebrew가 이미 설치되어 있습니다."
        return
    fi

    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "macOS 환경 감지됨. Homebrew 설치를 시작합니다..."

        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
            log_error "Homebrew 설치 실패"
            exit 1
        }

        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
        eval "$(/opt/homebrew/bin/brew shellenv)"

        if [ -f "$HOME/.zprofile" ]; then
            source "$HOME/.zprofile"
            log_info ".zprofile 적용 완료"
        fi

        log_info "macOS용 Homebrew 설치 완료"

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Ubuntu 환경 감지됨. Homebrew 설치를 시작합니다..."

        NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
            log_error "Homebrew 설치 실패"
            exit 1
        }

        echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> "$HOME/.profile"
        eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

        if [ -f "$HOME/.profile" ]; then
            source "$HOME/.profile"
            log_info ".profile 적용 완료"
        fi

        log_info "Ubuntu용 Homebrew 설치 완료"

    else
        log_warn "지원하지 않는 운영체제입니다: $OSTYPE"
    fi
}
setup_python_path() {
    log_info "Python PATH 설정 중..."

    PYTHON_BIN="$(brew --prefix)/bin"

    # bash
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q "$PYTHON_BIN" "$HOME/.bashrc"; then
            echo "export PATH=\"$PYTHON_BIN:\$PATH\"" >> "$HOME/.bashrc"
            log_info "~/.bashrc에 Python PATH 추가됨"
        fi
    fi

    # zsh
    if [ -f "$HOME/.zprofile" ]; then
        if ! grep -q "$PYTHON_BIN" "$HOME/.zprofile"; then
            echo "export PATH=\"$PYTHON_BIN:\$PATH\"" >> "$HOME/.zprofile"
            log_info "~/.zprofile에 Python PATH 추가됨"
        fi
    fi

    # 현재 세션에 적용
    export PATH="$PYTHON_BIN:$PATH"
    log_info "현재 세션에 Python PATH 적용됨: $PYTHON_BIN"

    # 확인
    if command -v python3.11 &>/dev/null; then
        python3.11 --version
    else
        log_warn "python3.11 명령어를 찾을 수 없습니다. 쉘 재시작 또는 source ~/.bashrc 권장"
    fi
}


install_python() {
    if ! command -v python3.11 &> /dev/null; then
        log_info "Python 3.11 설치 시작..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install python@3.11 || { log_error "Python 설치 실패"; exit 1; }
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt update && sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev || {
                log_error "Python 설치 실패"; exit 1;
            }
        else
            log_error "지원하지 않는 OS"; exit 1
        fi
        log_info "Python 3.11 설치 완료"
    else
        log_info "Python 3.11 이미 설치됨"
    fi
    python3.11 --version
}

install_poetry() {
    if ! command -v poetry &> /dev/null; then
        log_info "Poetry 설치 중..."
        curl -sSL https://install.python-poetry.org | python3 - || {
            log_error "Poetry 설치 실패"; exit 1;
        }

        export PATH="$HOME/.local/bin:$PATH"

        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            log_info "Poetry PATH를 .bashrc에 추가함"
        fi

        # 셸 설정 적용
        if [[ "$SHELL" == *"zsh" ]]; then
            [ -f "$HOME/.zprofile" ] && source "$HOME/.zprofile"
        elif [[ "$SHELL" == *"bash" ]]; then
            [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
        fi

        log_info "Poetry 설치 완료 및 PATH 적용"
    else
        log_info "Poetry가 이미 설치되어 있습니다."
    fi

    poetry --version
}


setup_project_files() {
    cd "$SCRIPT_DIR" || { log_error "디렉토리 이동 실패"; exit 1; }

    [ ! -f "pyproject.toml" ] && cat > pyproject.toml << EOF
[project]
name = "dev"
version = "0.1.0"
description = ""
authors = [{name = "phur99",email = "skerrigen12@gmail.com"}]
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
    "uvicorn (>=0.34.2,<0.35.0)"
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
torch = "==2.6.0"
transformers =">=4.52.3,<5.0.0"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
EOF

    [ ! -f "README.md" ] && cat > README.md << EOF
# dev

이 프로젝트는 Poetry 기반 Python 프로젝트입니다.

## 설치

```bash
init.sh
```

## 설명

FastAPI, vLLM, Qdrant 등을 사용하여 AI 파이프라인을 구축합니다.
EOF

    [ ! -f "config/settings.py" ] && mkdir -p config && echo -e "DATASET_NAME = \"squad\"\nDATASET_CACHE_DIR = \"${SCRIPT_DIR}/datasets\"" > config/settings.py
    touch config/__init__.py
}

install_dependencies() {
    local groups=("$@")
    poetry lock || { log_error "lock 파일 생성 실패"; exit 1; }

    if [ ${#groups[@]} -eq 0 ]; then
        # 기본 그룹(main)만 설치
        log_info "기본 의존성(main) 설치 시작..."
        poetry install --no-interaction --no-root --only main || {
            log_error "main 의존성 설치 실패"; exit 1;
        }
    else
        for group in "${groups[@]}"; do
            log_info "의존성 설치 시작: $group"
            poetry install --no-interaction --no-root --only "$group" || {
                log_error "$group 의존성 설치 실패"; exit 1;
            }
        done
    fi
    log_info "의존성 설치 완료"
}

install_ollama() {
    if ! command -v ollama &> /dev/null; then
        log_info "Ollama 설치 시작..."
        curl -fsSL https://ollama.com/install.sh | sh || {
            log_error "Ollama 설치 실패"; exit 1;
        }
    fi
    log_info "Ollama 서버 시작"
    ollama serve &
    OLLAMA_PID=$!
    sleep 5
    ollama pull bge-m3 || {
        log_error "모델 다운로드 실패"; kill $OLLAMA_PID; exit 1;
    }
    kill $OLLAMA_PID
    unset OLLAMA_PID
    log_info "Ollama 종료"
}

download_dataset() {
    log_info "데이터셋 다운로드 시작..."
    poetry run python "${SCRIPT_DIR}/scripts/download_dataset.py"
    log_info "데이터셋 다운로드 완료"
}

download_model() {
    log_info "모델 다운로드 시작..."
    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        read -p "HuggingFace 토큰을 입력하세요 (없으면 Enter): " hf_token
        echo "HF_TOKEN=${hf_token}" > "${SCRIPT_DIR}/.env"
        log_info ".env 파일 생성 완료"
    fi
    poetry run python "${SCRIPT_DIR}/scripts/download_model.py"
    log_info "모델 다운로드 완료"
}
start_docker_desktop_if_needed() {
    log_info "Docker 데몬 상태 확인 중..."

    if ! docker info &>/dev/null; then
        log_warn "Docker가 실행되고 있지 않습니다. Docker Desktop을 시작합니다..."

        open -a "Docker" || {
            log_error "Docker Desktop 실행 실패. 수동으로 실행해주세요."
            exit 1
        }

        log_info "Docker Desktop 실행 대기 중..."
        while ! docker info &>/dev/null; do
            sleep 2
        done

        log_info "Docker가 성공적으로 실행되었습니다."
    else
        log_info "Docker가 이미 실행 중입니다."
    fi
}

install_qdrant() {
    log_info "Docker 설치 여부 확인 중..."
    start_docker_desktop_if_needed
    if ! command -v docker &> /dev/null; then
        log_warn "Docker가 설치되어 있지 않습니다. 설치를 진행합니다."

        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install --cask docker || {
                log_error "Docker 설치 실패 (macOS)"
                exit 1
            }
            log_info "Docker Desktop 실행 후 로그인을 완료해야 Qdrant가 정상 작동합니다."
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt update
            sudo apt install -y ca-certificates curl gnupg
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            echo \
              "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] \
              https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
              sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt update
            sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || {
                log_error "Docker 설치 실패 (Linux)"
                exit 1
            }
        else
            log_error "지원하지 않는 OS입니다."
            exit 1
        fi
        log_info "Docker 설치 완료"
    else
        log_info "Docker가 이미 설치되어 있습니다."
    fi

    # Qdrant 컨테이너 실행
    log_info "Qdrant 서버를 Docker로 실행합니다..."

    if docker ps -a --format '{{.Names}}' | grep -q "^qdrant$"; then
        log_warn "기존 qdrant 컨테이너가 이미 존재합니다. 설치를 건너뜁니다."
        return 0
    fi
    docker run -d --name qdrant --restart unless-stopped \
        -p 6333:6333 \
        -p 6334:6334 \
        qdrant/qdrant || {
            log_error "Qdrant 실행 실패"
            exit 1
        }

    log_info "Qdrant 서버가 http://localhost:6333 에서 실행 중입니다."
    poetry run python "${SCRIPT_DIR}/scripts/make_db.py"
}


# 실행
init "$@"