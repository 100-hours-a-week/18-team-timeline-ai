import requests
from bs4 import BeautifulSoup


def get_img_link(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        og_image = soup.find("meta", property="og:image")

        if og_image and og_image.get("content"):
            return og_image["content"]
        else:
            return ""  # 이미지 링크가 없을 경우 빈 문자열 반환
    except Exception as e:
        print(f"이미지 추출 실패 ({url}): {e}")
        return ""
