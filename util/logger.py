import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class JsonFormatter(logging.Formatter):
    """JSON 형식으로 로그를 포맷팅하는 클래스"""

    def format(self, record: logging.LogRecord) -> str:
        """로그 레코드를 JSON 형식으로 변환합니다.

        Args:
            record: 로그 레코드

        Returns:
            str: JSON 형식의 로그 문자열
        """
        # 기본 로그 데이터
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 예외 정보가 있는 경우 추가
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # 추가 속성이 있는 경우 추가
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
    """JSON 형식의 로그를 관리하는 클래스"""

    _instances: Dict[str, "Logger"] = {}

    def __new__(
        cls, name: str, log_level: int = logging.INFO, log_dir: str = "logs"
    ) -> "Logger":
        """싱글톤 패턴으로 로거 인스턴스 관리

        Args:
            name: 로거 이름
            log_level: 로그 레벨
            log_dir: 로그 디렉토리

        Returns:
            Logger: 로거 인스턴스
        """
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(
        self, name: str, log_level: int = logging.INFO, log_dir: str = "logs"
    ) -> None:
        """Logger 초기화

        Args:
            name: 로거 이름
            log_level: 로그 레벨
            log_dir: 로그 디렉토리
        """
        if not hasattr(self, "initialized"):
            self.name = name
            self.log_level = log_level
            self.log_dir = PROJECT_ROOT / log_dir
            self.logger = self._setup_logger()
            self.initialized = True

    def _setup_logger(self) -> logging.Logger:
        """로거 설정

        Returns:
            logging.Logger: 설정된 로거 인스턴스
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)

        if logger.handlers:
            return logger

        # 로그 디렉토리 생성
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_path = self.log_dir / current_date
        log_path.mkdir(parents=True, exist_ok=True)

        # 로그 파일명 생성
        log_file = log_path / f"{self.name}_{current_date}.json"

        # 파일 핸들러 설정
        file_handler = logging.FileHandler(filename=str(log_file), encoding="utf-8")
        file_handler.setLevel(self.log_level)

        # 콘솔 핸들러 설정 (ERROR 레벨만)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)

        # JSON 포맷터 설정
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        console_handler.setFormatter(json_formatter)

        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log(
        self,
        level: int,
        message: str,
        **kwargs: Any,
    ) -> None:
        """로그를 기록합니다.

        Args:
            level: 로그 레벨
            message: 로그 메시지
            **kwargs: 추가 로그 데이터
        """
        extra = kwargs.copy()
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """디버그 레벨 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 로그 데이터
        """
        self.log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """정보 레벨 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 로그 데이터
        """
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """경고 레벨 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 로그 데이터
        """
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """에러 레벨 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 로그 데이터
        """
        self.log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """치명적 에러 레벨 로그를 기록합니다.

        Args:
            message: 로그 메시지
            **kwargs: 추가 로그 데이터
        """
        self.log(logging.CRITICAL, message, **kwargs)

    @classmethod
    def get_logger(
        cls, name: str, log_level: int = logging.INFO, log_dir: str = "logs"
    ) -> "Logger":
        """로거 인스턴스를 가져옵니다.

        Args:
            name: 로거 이름
            log_level: 로그 레벨
            log_dir: 로그 디렉토리

        Returns:
            Logger: 로거 인스턴스
        """
        return cls(name, log_level, log_dir)


def setup_root_logger(log_level: int = logging.INFO, log_dir: str = "logs") -> None:
    """루트 로거를 설정합니다.

    Args:
        log_level: 로그 레벨
        log_dir: 로그 디렉토리
    """
    Logger.get_logger("root", log_level, log_dir)
