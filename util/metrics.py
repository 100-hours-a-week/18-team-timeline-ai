from typing import Dict, Any, Optional, Protocol, runtime_checkable, TypeVar, Generic
from abc import ABC, abstractmethod
import time
from datetime import datetime
import json
from dataclasses import dataclass, field
from enum import Enum

T = TypeVar("T")


class MetricType(Enum):
    """메트릭 유형 정의"""

    COUNTER = "counter"  # 증가만 하는 카운터
    GAUGE = "gauge"  # 증가/감소 가능한 게이지
    HISTOGRAM = "histogram"  # 분포를 측정하는 히스토그램
    TIMER = "timer"  # 시간 측정


@dataclass
class MetricValue(Generic[T]):
    """메트릭 값 데이터 클래스"""

    value: T
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "timestamp": self.timestamp, "labels": self.labels}


@runtime_checkable
class MetricStorage(Protocol[T]):
    """메트릭 저장소 프로토콜"""

    def store(self, metric_name: str, value: MetricValue[T]) -> None:
        """메트릭 저장"""
        ...

    def retrieve(self, metric_name: str) -> Optional[MetricValue[T]]:
        """메트릭 조회"""
        ...

    def list_metrics(self) -> Dict[str, Any]:
        """모든 메트릭 조회"""
        ...


class InMemoryMetricStorage(MetricStorage[float]):
    """메모리 기반 메트릭 저장소"""

    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}

    def store(self, metric_name: str, value: MetricValue[float]) -> None:
        if metric_name not in self._storage:
            self._storage[metric_name] = {"values": []}
        self._storage[metric_name]["values"].append(value)

    def retrieve(self, metric_name: str) -> Optional[MetricValue[float]]:
        if metric_name in self._storage and self._storage[metric_name]["values"]:
            return self._storage[metric_name]["values"][-1]
        return None

    def list_metrics(self) -> Dict[str, Any]:
        return self._storage


class MetricCollector(ABC):
    """메트릭 수집기 추상 클래스"""

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """메트릭 수집"""
        pass


class MetricAggregator(ABC):
    """메트릭 집계기 추상 클래스"""

    @abstractmethod
    def aggregate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """메트릭 집계"""
        pass


class BaseMetrics(Generic[T]):
    """기본 메트릭 클래스"""

    def __init__(
        self,
        name: str,
        storage: Optional[MetricStorage[T]] = None,
        collectors: Optional[list[MetricCollector]] = None,
        aggregators: Optional[list[MetricAggregator]] = None,
    ):
        self.name = name
        self.storage = storage or InMemoryMetricStorage()
        self.collectors = collectors or []
        self.aggregators = aggregators or []
        self.start_time = time.time()
        self.metric_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_metric(
        self,
        name: str,
        value: T,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """메트릭 추가"""
        metric_value = MetricValue(value=value, labels=labels or {})
        self.storage.store(name, metric_value)

    def increment(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """카운터 증가"""
        current_value = 0.0
        if last_value := self.storage.retrieve(name):
            current_value = last_value.value
        self.add_metric(name, current_value + value, MetricType.COUNTER, labels)

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """게이지 값 설정"""
        self.add_metric(name, value, MetricType.GAUGE, labels)

    def record_time(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """시간 측정 기록"""
        self.add_metric(name, duration, MetricType.TIMER, labels)

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """히스토그램 값 기록"""
        self.add_metric(name, value, MetricType.HISTOGRAM, labels)

    def get_metric(self, name: str) -> Optional[MetricValue[T]]:
        """메트릭 조회"""
        return self.storage.retrieve(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """모든 메트릭 조회"""
        metrics = self.storage.list_metrics()
        collected_metrics = {}

        # 수집기 실행
        for collector in self.collectors:
            collected_metrics.update(collector.collect())

        # 집계기 실행
        aggregated_metrics = metrics
        for aggregator in self.aggregators:
            aggregated_metrics = aggregator.aggregate(aggregated_metrics)

        return {
            "name": self.name,
            "id": self.metric_id,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time,
            "metrics": aggregated_metrics,
            "collected_metrics": collected_metrics,
        }

    def to_json(self) -> str:
        """메트릭을 JSON 문자열로 변환"""
        return json.dumps(self.get_all_metrics(), default=str)


class Timer:
    """시간 측정을 위한 컨텍스트 매니저"""

    def __init__(
        self,
        metrics: BaseMetrics[float],
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.metrics.record_time(self.name, duration, self.labels)


# 예시: 시스템 리소스 수집기
class SystemResourceCollector(MetricCollector):
    """시스템 리소스 수집기"""

    def collect(self) -> Dict[str, Any]:
        import psutil

        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        }


# 예시: 커스텀 메트릭 클래스
class SentimentMetrics(BaseMetrics):
    """감정 분석 메트릭"""

    def __init__(self):
        super().__init__(
            name="sentiment_analysis", collectors=[SystemResourceCollector()]
        )
        self.total_processed = 0
        self.success_count = 0
        self.error_count = 0

    def record_sentiment(self, sentiment: str, score: float) -> None:
        """감정 분석 결과 기록"""
        self.increment(f"sentiment_{sentiment}")
        self.record_histogram(f"sentiment_score_{sentiment}", score)

    def record_processing_result(self, success: bool) -> None:
        """처리 결과 기록"""
        self.total_processed += 1
        if success:
            self.success_count += 1
            self.increment("success_count")
        else:
            self.error_count += 1
            self.increment("error_count")


# 사용 예시
if __name__ == "__main__":
    # 기본 메트릭 사용
    metrics = BaseMetrics("test_metrics")
    metrics.increment("request_count")
    metrics.set_gauge("active_connections", 10)

    # 감정 분석 메트릭 사용
    sentiment_metrics = SentimentMetrics()
    with Timer(sentiment_metrics, "processing_time"):
        sentiment_metrics.record_sentiment("positive", 0.8)
        sentiment_metrics.record_processing_result(True)

    print(sentiment_metrics.to_json())
