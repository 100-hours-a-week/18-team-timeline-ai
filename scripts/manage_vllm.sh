#!/bin/bash

# 로깅 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 서버 상태 확인
check_status() {
    if [ -f "vllm_server.pid" ]; then
        pid=$(cat vllm_server.pid)
        if ps -p $pid > /dev/null; then
            log "✅ vLLM 서버가 실행 중입니다 (PID: $pid)"
            return 0
        else
            log "❌ vLLM 서버가 비정상 종료되었습니다"
            rm -f vllm_server.pid
            return 1
        fi
    else
        log "❌ vLLM 서버가 실행되지 않고 있습니다"
        return 1
    fi
}

# 서버 시작
start_server() {
    if check_status; then
        log "이미 vLLM 서버가 실행 중입니다"
        return
    fi

    log "🚀 vLLM 서버를 시작합니다..."
    nohup python3 scripts/run_vllm_server.py > /dev/null 2>&1 &
    
    # 서버가 시작될 때까지 대기
    for i in {1..30}; do
        if check_status; then
            log "✅ vLLM 서버가 성공적으로 시작되었습니다"
            return 0
        fi
        sleep 1
    done
    
    log "❌ vLLM 서버 시작 실패"
    return 1
}

# 서버 중지
stop_server() {
    if [ -f "vllm_server.pid" ]; then
        pid=$(cat vllm_server.pid)
        log "🛑 vLLM 서버를 중지합니다 (PID: $pid)..."
        kill $pid
        rm -f vllm_server.pid
        log "✅ vLLM 서버가 중지되었습니다"
    else
        log "vLLM 서버가 실행되지 않고 있습니다"
    fi
}

# 서버 재시작
restart_server() {
    log "🔄 vLLM 서버를 재시작합니다..."
    stop_server
    sleep 2
    start_server
}

# 명령어 처리
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        check_status
        ;;
    *)
        echo "사용법: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

exit 0 