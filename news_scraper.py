# improved_news_scraper.py
import requests
from bs4 import BeautifulSoup
import feedparser
import time
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImprovedNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 검증된 RSS 피드 목록
        self.verified_sources = {
            "MIT Technology Review": "https://www.technologyreview.com/feed/",
            "Wired": "https://www.wired.com/feed/rss",
            "TechCrunch": "https://techcrunch.com/feed/",
            "The Verge": "https://www.theverge.com/rss/index.xml",
            "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
            "IEEE Spectrum": "https://spectrum.ieee.org/feeds/topic/computing.rss",
            "Hacker News": "https://hnrss.org/frontpage",
            "TechRadar": "https://www.techradar.com/rss",
            "ZDNet": "https://www.zdnet.com/news/rss.xml",
            "Engadget": "https://www.engadget.com/rss.xml"
        }
    
    def test_rss_feed(self, url: str) -> bool:
        """RSS 피드가 작동하는지 테스트"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # RSS 피드인지 확인
            feed = feedparser.parse(response.content)
            return len(feed.entries) > 0
            
        except Exception as e:
            logger.warning(f"RSS 피드 테스트 실패 {url}: {e}")
            return False
    
    def get_working_feeds(self) -> dict:
        """작동하는 RSS 피드만 반환"""
        working_feeds = {}
        
        for source_name, rss_url in self.verified_sources.items():
            logger.info(f"RSS 피드 테스트 중: {source_name}")
            
            if self.test_rss_feed(rss_url):
                working_feeds[source_name] = rss_url
                logger.info(f"✓ 작동: {source_name}")
            else:
                logger.warning(f"✗ 실패: {source_name}")
            
            time.sleep(random.uniform(1, 2))
        
        return working_feeds
    
    def scrape_rss_with_retry(self, url: str, source_name: str, max_retries: int = 3) -> list:
        """재시도 로직이 있는 RSS 스크래핑"""
        articles = []
        
        for attempt in range(max_retries):
            try:
                feed = feedparser.parse(url)
                
                if not feed.entries:
                    raise Exception("RSS 피드가 비어있음")
                
                for entry in feed.entries[:10]:
                    article = {
                        'title': entry.get('title', ''),
                        'content': entry.get('summary', '') or entry.get('description', ''),
                        'url': entry.get('link', ''),
                        'source': source_name,
                        'published_at': entry.get('published', ''),
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    # 빈 제목이나 URL이 없는 기사는 제외
                    if article['title'] and article['url']:
                        articles.append(article)
                
                logger.info(f"✓ {source_name}: {len(articles)}개 기사 수집")
                return articles
                
            except Exception as e:
                logger.warning(f"시도 {attempt + 1}/{max_retries} 실패 - {source_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 5))
                else:
                    logger.error(f"최종 실패: {source_name}")
        
        return articles
    
    def scrape_all_working_sources(self) -> list:
        """작동하는 모든 소스에서 뉴스 수집"""
        all_articles = []
        
        # 먼저 작동하는 피드 확인
        working_feeds = self.get_working_feeds()
        
        if not working_feeds:
            logger.error("작동하는 RSS 피드가 없습니다!")
            return []
        
        # 각 피드에서 기사 수집
        for source_name, rss_url in working_feeds.items():
            logger.info(f"기사 수집 중: {source_name}")
            articles = self.scrape_rss_with_retry(rss_url, source_name)
            all_articles.extend(articles)
            
            # 요청 간격 조절
            time.sleep(random.uniform(2, 4))
        
        # 중복 제거
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            if article['url'] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article['url'])
        
        logger.info(f"총 {len(unique_articles)}개의 고유 기사 수집 완료")
        return unique_articles
    
    def save_articles(self, articles: list, filename: str = "scraped_news.json"):
        """기사 저장"""
        import json
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"기사 저장 완료: {filename}")

# 사용 예시
if __name__ == "__main__":
    scraper = ImprovedNewsScraper()
    
    # 뉴스 수집
    articles = scraper.scrape_all_working_sources()
    
    # 결과 저장
    scraper.save_articles(articles)
    
    # 결과 출력
    print(f"\n=== 수집 결과 ===")
    print(f"총 기사 수: {len(articles)}")
    
    for i, article in enumerate(articles[:5]):
        print(f"\n{i+1}. {article['title']}")
        print(f"   소스: {article['source']}")
        print(f"   URL: {article['url']}")
        print(f"   내용: {article['content'][:100]}...")
