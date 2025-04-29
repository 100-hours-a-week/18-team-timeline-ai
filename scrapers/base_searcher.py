from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class BaseSearcher(ABC):
    """검색기 인터페이스"""

    @abstractmethod
    def search(self, input_data: Any) -> Any:
        pass
