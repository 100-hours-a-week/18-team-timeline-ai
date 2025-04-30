from pydantic import BaseModel
from datetime import datetime
from typing import List, Generic, TypeVar
from models.timeline_card import TimelineCard

# -----------------------------------------------

T = TypeVar("T")


class CommonResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str
    data: T


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


# -----------------------------------------------


# Request
class TimelineRequest(BaseModel):
    query: List[str]
    startAt: datetime
    endAt: datetime


# Response - CommonResponse[TimelineData]
class TimelineData(BaseModel):
    title: str
    summary: str
    image: str
    category: str  # enum인데 나중에 확인 필요
    timeline: List[TimelineCard]


# -----------------------------------------------


# Request
class MergeRequest(BaseModel):
    timeline: List[TimelineCard]


# Response - CommonResponse[TimelineCard]

# -----------------------------------------------


# Request
class HotRequest(BaseModel):
    num: int


# Response - CommonResponse[HotData]
class HotData(BaseModel):
    keywords: List[str]


# -----------------------------------------------


# Request
class CommentRequest(BaseModel):
    query: List[str]
    num: int


# Response - CommonResponse[CommentData]
class CommentData(BaseModel):
    positive: int
    neutral: int
    negative: int


# -----------------------------------------------
