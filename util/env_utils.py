import os
import random
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
    if not SERPER_API_KEYS or i < 0:
        return ""

    SERPER_API_KEYS = SERPER_API_KEYS.split(",")
    if i >= len(SERPER_API_KEYS):
        return ""

    return random.choice(SERPER_API_KEYS).strip()
