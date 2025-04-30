import dotenv
import os
import asyncio
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher


async def main():
    dotenv.load_dotenv(override=True)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    REST_API_KEY = os.getenv("REST_API_KEY")
    video_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
    youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY)
    df = video_searcher.search("")
    ripple = await youtube_searcher.search(df=df)
    print(*ripple)


if __name__ == "__main__":
    asyncio.run(main=main())
