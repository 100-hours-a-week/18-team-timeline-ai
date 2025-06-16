import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON 형식의 로그 포맷터"""

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형식으로 변환

        Args:
            record (logging.LogRecord): 로그 레코드

        Returns:
            str: JSON 형식의 로그 문자열
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # 예외 정보가 있는 경우 추가
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 추가 속성이 있는 경우 포함
        for key, value in record.__dict__.items():
            if key not in [
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            ]:
                log_data[key] = value

        return json.dumps(log_data, ensure_ascii=False)


class Logger:
    """로깅을 관리하는 클래스"""

    _instances: dict = {}

    def __new__(
        cls, name: str, log_level: int = logging.ERROR, log_dir: str = "logs"
    ) -> "Logger":
        """싱글톤 패턴으로 로거 인스턴스 관리

        Args:
            name (str): 로거 이름
            log_level (int, optional): 로그 레벨. Defaults to logging.ERROR.
            log_dir (str, optional): 로그 파일이 저장될 디렉토리.
                Defaults to "logs".

        Returns:
            Logger: 로거 인스턴스
        """
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(
        self, name: str, log_level: int = logging.ERROR, log_dir: str = "logs"
    ) -> None:
        """Logger 초기화

        Args:
            name (str): 로거 이름
            log_level (int, optional): 로그 레벨. Defaults to logging.ERROR.
            log_dir (str, optional): 로그 파일이 저장될 디렉토리.
                Defaults to "logs".
        """
        if not hasattr(self, "initialized"):
            self.name = name
            self.log_level = log_level
            self.log_dir = log_dir
            self.logger = self._setup_logger()
            self.error_logger = self._setup_error_logger()
            self.initialized = True

    def _setup_logger(self) -> logging.Logger:
        """일반 로거 설정

        Returns:
            logging.Logger: 설정된 로거 인스턴스
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)

        if logger.handlers:
            return logger

        # 로그 디렉토리 생성
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_path = Path(self.log_dir) / current_date
        log_path.mkdir(parents=True, exist_ok=True)

        # 로그 파일명 생성
        log_file = log_path / f"{self.name}_{current_date}.log"

        # 파일 핸들러 설정
        file_handler = logging.FileHandler(filename=log_file, encoding="utf-8")
        file_handler.setLevel(self.log_level)

        # JSON 포맷터 설정
        json_formatter = JSONFormatter()
        file_handler.setFormatter(json_formatter)

        # 핸들러 추가
        logger.addHandler(file_handler)

        return logger

    def _setup_error_logger(self) -> logging.Logger:
        """에러 전용 로거 설정

        Returns:
            logging.Logger: 설정된 에러 로거 인스턴스
        """
        error_logger = logging.getLogger(f"{self.name}_error")
        error_logger.setLevel(logging.ERROR)

        if error_logger.handlers:
            return error_logger

        # 로그 디렉토리 생성
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_path = Path(self.log_dir) / current_date
        log_path.mkdir(parents=True, exist_ok=True)

        # 에러 로그 파일명 생성
        error_log_file = log_path / f"{self.name}_error_{current_date}.log"

        # 에러 파일 핸들러 설정
        error_file_handler = logging.FileHandler(
            filename=error_log_file, encoding="utf-8"
        )
        error_file_handler.setLevel(logging.ERROR)

        # 콘솔 핸들러 설정 (에러만 출력)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)

        # JSON 포맷터 설정
        json_formatter = JSONFormatter()
        error_file_handler.setFormatter(json_formatter)
        console_handler.setFormatter(json_formatter)

        # 핸들러 추가
        error_logger.addHandler(error_file_handler)
        error_logger.addHandler(console_handler)

        return error_logger

    def debug(self, message: str) -> None:
        """디버그 레벨 로그 기록

        Args:
            message (str): 로그 메시지
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """정보 레벨 로그 기록

        Args:
            message (str): 로그 메시지
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """경고 레벨 로그 기록

        Args:
            message (str): 로그 메시지
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """에러 레벨 로그 기록

        Args:
            message (str): 로그 메시지
        """
        self.logger.error(message)
        self.error_logger.error(message)

    def critical(self, message: str) -> None:
        """치명적 에러 레벨 로그 기록

        Args:
            message (str): 로그 메시지
        """
        self.logger.critical(message)
        self.error_logger.critical(message)

    @classmethod
    def get_logger(
        cls, name: str, log_level: int = logging.ERROR, log_dir: str = "logs"
    ) -> "Logger":
        """로거 인스턴스를 가져옵니다.

        Args:
            name (str): 로거 이름
            log_level (int, optional): 로그 레벨. Defaults to logging.ERROR.
            log_dir (str, optional): 로그 파일이 저장될 디렉토리.
                Defaults to "logs".

        Returns:
            Logger: 로거 인스턴스
        """
        return cls(name, log_level, log_dir)


def setup_root_logger(log_level: int = logging.INFO, log_dir: str = "logs") -> None:
    """루트 로거를 설정합니다.

    Args:
        log_level (int, optional): 로그 레벨. Defaults to logging.INFO.
        log_dir (str, optional): 로그 파일이 저장될 디렉토리.
            Defaults to "logs".
    """
    Logger.get_logger("root", log_level, log_dir)
