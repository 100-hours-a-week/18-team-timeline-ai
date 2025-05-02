from datetime import datetime, timedelta
import re


def parse_relative_date(date_str: str) -> datetime:
    now = datetime.now()

    if "분 전" in date_str:
        minutes = int(re.search(r"\d+", date_str).group())
        return now - timedelta(minutes=minutes)
    elif "시간 전" in date_str:
        hours = int(re.search(r"\d+", date_str).group())
        return now - timedelta(hours=hours)
    elif "일 전" in date_str:
        days = int(re.search(r"\d+", date_str).group())
        return now - timedelta(days=days)
    elif "주 전" in date_str:
        weeks = int(re.search(r"\d+", date_str).group())
        return now - timedelta(weeks=weeks)
    elif "개월 전" in date_str:
        months = int(re.search(r"\d+", date_str).group())
        return now - timedelta(days=30 * months)
    else:
        # 예외적으로 날짜 포맷이 이상하면 현재로
        return now
