from pydantic import BaseModel
from datetime import datetime
from typing import List


class TimelineCard(BaseModel):
    title: str
    content: str
    startAt: datetime
    endAt: datetime
    source: List[str]


"""
card = TimelineCard(
    title="우크라이나 전쟁 격화",
    content="러시아군과 우크라이나군의 충돌이 심화되고 있습니다.",
    startAt=datetime(2025, 4, 1),
    endAt=datetime(2025, 4, 7),
    source=[
        "https://news.example.com/1",
        "https://news.example.com/2"
    ]
)
"""
