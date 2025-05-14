from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import List


class TimelineCard(BaseModel):
    title: str
    content: str
    duration: str
    startAt: datetime
    endAt: datetime
    source: List[str]

    @field_validator('startAt', 'endAt', mode='before')
    @classmethod
    def parse_date_or_datetime(cls, value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # ISO 8601 full datetime
                return datetime.fromisoformat(value)
            except ValueError:
                pass
            try:
                # Date only → add midnight time
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                pass
        raise ValueError(f"Invalid date/datetime format: {value}")

"""
card = TimelineCard(
    title="우크라이나 전쟁 격화",
    content="러시아군과 우크라이나군의 충돌이 심화되고 있습니다.",
    duration: "WEEK",
    startAt=datetime(2025, 4, 1),
    endAt=datetime(2025, 4, 7),
    source=[
        "https://news.example.com/1",
        "https://news.example.com/2"
    ]
)
"""
