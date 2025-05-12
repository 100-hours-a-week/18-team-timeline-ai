class NoSearchResultError(ValueError):
    """404: 검색 결과가 없을 때 발생하는 에러"""

    pass


class InvalidAPIKeyError(ValueError):
    """API Key가 잘못되었을 때 발생하는 에러"""

    pass


class EnvVarNotFoundError(EnvironmentError):
    """필수 환경변수가 누락되었을 때 발생하는 에러"""

    pass


class InvalidQueryError(ValueError):
    """400: Query의 형식이 잘못되었을 때 발생하는 에러"""

    pass


class SearchRequestFailedError(ValueError):
    """검색 토큰이 부족하거나 api 서버에 문제가 발생 시 발생하는 에러"""

    pass
