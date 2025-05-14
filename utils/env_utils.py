import os
from dotenv import load_dotenv


def get_serp_key(i: int):
    load_dotenv()
    SERP_API_KEYS = os.getenv("SERP_API_KEYS")
    if not SERP_API_KEYS:
        return ""
    SERP_API_KEYS = SERP_API_KEYS.split(",")
    return SERP_API_KEYS[i].strip()


def get_serper_key(i: int):
    load_dotenv()
    SERPER_API_KEYS = os.getenv("SERPER_API_KEYS")
    if not SERPER_API_KEYS:
        return ""
    SERPER_API_KEYS = SERPER_API_KEYS.split(",")
    return SERPER_API_KEYS[i].strip()
