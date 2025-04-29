from PyKakao import DaumSearch
import dotenv
import pandas as pd
from pprint import pprint
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import re, os
from pathlib import Path
from typing import Optional


class DaumVclipSearcher:
    """Daum Vclip 검색기"""

    def __init__(self, path: Optional[str] = None):
        """초기화 작업 수행

        Args:
            path (Optional[Path], optional): .env의 경로 지정. 기본값: None

        Raises:
            ValueError: REST_API_KEY를 환경변수 목록에서 찾아올 수 없을 경우 반환
        """
        if path:
            dotenv.load_dotenv(dotenv_path=path, override=True)
        else:
            dotenv.load_dotenv(override=True)
        rest_api_key = os.getenv("REST_API_KEY")
        if not rest_api_key:
            raise ValueError("REST_API_KEY를 환경변수 목록에서 찾을 수 없습니다.")

        self.api = DaumSearch(service_key=rest_api_key)

    def search(self, query: str, page: int = 2) -> pd.DataFrame:
        """비디오 클립 검색 후 목록을 반환

        Args:
            query (str): 검색 문자열
            page (int, optional): 검색하고 싶은 페이지. Defaults to 2.

        Raises:
            ValueError: 검색어에 대한 영상을 찾을 수 없을 경우 반환
        Returns:
            pd.DataFrame: {author, datetime, play_time, thumbnail, title, url}을 column으로 가지는 DataFrame 객체 반환
        """
        df = self.api.search_vclip(
            query=query, page=page, dataframe=True, data_type="json"
        )
        if df.empty:
            raise ValueError(f"{query}에 대한 영상을 찾을 수 없습니다.")
        return df


def main():
    # 보통은 main 함수에서 시작
    searcher = DaumVclipSearcher()
    df = searcher.search("")
    pprint(df)


if __name__ == "__main__":
    main()
