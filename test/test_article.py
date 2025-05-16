from scrapers.article_extractor import ArticleExtractor


def main():
    extractor = ArticleExtractor()
    URLS = [
        "https://www.hani.co.kr/arti/society/society_general/1192251.html",
        "https://www.hani.co.kr/arti/society/society_general/1192255.html",
        "https://www.hankyung.com/article/2025041493977",
        "https://www.khan.co.kr/article/202504141136001",
        "https://www.mk.co.kr/news/politics/11290687",
        "https://www.chosun.com/politics/politics_general/2025/04/14/THWVKUHQG5CKFJF6CLZLP5PKM4",
    ]
    results = extractor.search(URLS)
    for url, title, text in results:
        print(f"URL: {url}\nTitle: {title}\nText: {text[:100]}...\n")


if __name__ == "__main__":
    main()
