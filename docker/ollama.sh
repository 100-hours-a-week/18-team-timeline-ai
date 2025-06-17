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
