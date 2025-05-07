import subprocess
import threading
import time
import socket
import logging
import psutil
import sys
import os
import signal

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vllm_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# PID 파일 경로
PID_FILE = "vllm_server.pid"


def write_pid_file(pid):
    """PID 파일에 프로세스 ID를 기록하는 함수"""
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def read_pid_file():
    """PID 파일에서 프로세스 ID를 읽는 함수"""
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def remove_pid_file():
    """PID 파일을 삭제하는 함수"""
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass


def is_port_in_use(port, host="localhost"):
    """포트가 사용 중인지 확인하는 함수

    Args:
        port (int): 확인할 포트 번호
        host (str): 호스트 주소 (기본값: localhost)

    Returns:
        bool: 포트 사용 여부
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def is_vllm_running():
    """vLLM 서버 프로세스가 실행 중인지 확인하는 함수

    Returns:
        bool: 서버 실행 여부
    """
    pid = read_pid_file()
    if pid is None:
        return False

    try:
        process = psutil.Process(pid)
        return (
            process.is_running()
            and "vllm.entrypoints.openai.api_server" in " ".join(process.cmdline())
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def stream_output(process):
    """프로세스의 출력을 실시간으로 스트리밍하는 함수

    Args:
        process: 실행 중인 프로세스 객체
    """

    def stream(pipe):
        # 파이프에서 한 줄씩 읽어서 로그로 출력
        for line in iter(pipe.readline, b""):
            logger.info(line.decode("utf-8").rstrip())

    # stdout과 stderr를 별도의 스레드에서 처리
    # daemon=True로 설정하여 메인 스레드가 종료되면 함께 종료되도록 함
    threading.Thread(target=stream, args=(process.stdout,), daemon=True).start()
    threading.Thread(target=stream, args=(process.stderr,), daemon=True).start()


def wait_for_port(port, host="localhost"):
    """포트가 열릴 때까지 대기하는 함수

    Args:
        port (int): 확인할 포트 번호
        host (str): 호스트 주소 (기본값: localhost)
    """
    while True:
        try:
            # 포트 연결 시도
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            # 연결 실패시 1초 대기 후 재시도
            time.sleep(1)


def start_vllm_server():
    """vLLM 서버를 시작하는 함수

    Returns:
        subprocess.Popen: 실행된 프로세스 객체
    """
    # vLLM 서버 실행 명령어 설정
    vllm_cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "models/HyperCLOVAX-SEED-Text-Instruct-1.5B",  # 로컬 모델 경로
        "--trust-remote-code",  # 원격 코드 신뢰
        "--port",
        "8000",  # 서버 포트
        "--max-model-len",
        "8192",  # 최대 모델 길이
        "--gpu-memory-utilization",
        "0.9",  # GPU 메모리 사용률
        "--max-num-seqs",
        "5",  # 최대 시퀀스 수
        "--max-num-batched-tokens",
        "4096",  # 최대 배치 토큰 수
        "--disable-log-requests",  # 요청 로깅 비활성화
        "--dtype",
        "half",  # 데이터 타입 (FP16)
    ]

    # vLLM 서버 프로세스 시작
    logger.info("🚀 vLLM 서버를 시작합니다...")
    vllm_process = subprocess.Popen(
        vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # PID 파일에 프로세스 ID 기록
    write_pid_file(vllm_process.pid)

    # 로그 출력 스레드 시작
    stream_output(vllm_process)

    # 포트가 열릴 때까지 대기
    logger.info("⏳ 포트 8000이 열릴 때까지 대기 중...")
    wait_for_port(8000)
    logger.info("✅ vLLM 서버가 준비되었습니다!")

    return vllm_process


def monitor_server():
    """서버 상태를 모니터링하고 필요시 재시작하는 함수"""
    while True:
        try:
            # 서버 상태 확인
            if not is_vllm_running() or not is_port_in_use(8000):
                logger.warning("vLLM 서버가 실행되지 않고 있습니다. 재시작합니다...")
                start_vllm_server()
            else:
                logger.info("vLLM 서버가 정상적으로 실행 중입니다.")

            # 30초마다 상태 확인
            time.sleep(30)

        except KeyboardInterrupt:
            logger.info("🛑 모니터링을 종료합니다...")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ 오류 발생: {str(e)}")
            time.sleep(5)  # 오류 발생시 5초 대기 후 재시도


def main():
    """vLLM 서버를 실행하고 관리하는 메인 함수"""
    try:
        # 서버가 이미 실행 중인지 확인
        if is_vllm_running() and is_port_in_use(8000):
            logger.info("vLLM 서버가 이미 실행 중입니다.")
            return

        # 서버 시작
        start_vllm_server()

        # 서버 모니터링 시작
        monitor_server()

    except KeyboardInterrupt:
        # Ctrl+C로 종료시 정상적으로 프로세스 종료
        logger.info("🛑 서버를 종료합니다...")
        pid = read_pid_file()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        remove_pid_file()
    except Exception as e:
        # 예외 발생시 로그 기록 후 프로세스 종료
        logger.error(f"❌ 오류 발생: {str(e)}")
        pid = read_pid_file()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        remove_pid_file()
        raise


if __name__ == "__main__":
    main()
