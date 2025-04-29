import os
from fastapi import APIRouter
from dotenv import load_dotenv
from scrapers.serpapi import get_trending_keywords

router = APIRouter()

@router.post("/hot")
def get_hot_topics(count: int = 3) -> list[str]:
    load_dotenv()
    SERP_API_KEYS = os.getenv("SERP_API_KEYS")
    if not SERP_API_KEYS:
        raise ValueError("SERP_API_KEYS not found in .env file.")
    
    SERP_API_KEYS = SERP_API_KEYS.split(",")
    SERP_API_KEY = SERP_API_KEYS[0].strip()

    keywords = get_trending_keywords(SERP_API_KEY)
    count = min(count, len(keywords))
    return keywords[:count]
