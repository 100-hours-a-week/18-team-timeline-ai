from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseSearcher(ABC):
    """검색기 인터페이스

    모든 검색기 클래스의 기본이 되는 추상 클래스입니다.
    이 클래스를 상속받는 모든 클래스는 search 메서드를 구현해야 합니다.

    Attributes:
        None

    Methods:
        search: 검색을 수행하는 추상 메서드
    """

    @abstractmethod
    def search(self, input_data: Any) -> Any:
        """검색을 수행하는 추상 메서드

        Args:
            input_data (Any): 검색에 필요한 입력 데이터

        Returns:
            Any: 검색 결과

        Raises:
            NotImplementedError: 이 메서드를 구현하지 않은 경우 발생
        """
        pass
