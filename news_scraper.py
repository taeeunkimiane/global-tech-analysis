# news_scraper.py
import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import random
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Optional
import re
from textblob import TextBlob
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalTechNewsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 국가별 뉴스 소스 정의
        self.news_sources = {
            "미국": {
                "sources": [
                    {"name": "MIT Technology Review", "rss": "https://www.technologyreview.com/feed/"},
                    {"name": "Wired", "rss": "https://www.wired.com/feed/"},
                    {"name": "TechCrunch", "rss": "https://techcrunch.com/feed/"},
                    {"name": "IEEE Spectrum", "rss": "https://spectrum.ieee.org/rss/blog/tech-talk"},
                ]
            },
            "유럽": {
                "sources": [
                    {"name": "The Next Web", "rss": "https://thenextweb.com/feed/"},
                    {"name": "ZDNet Europe", "rss": "https://www.zdnet.com/news/rss.xml"},
                ]
            },
            "아시아": {
                "sources": [
                    {"name": "Nikkei Asia", "rss": "https://asia.nikkei.com/rss/Technology"},
                    {"name": "Tech in Asia", "rss": "https://www.techinasia.com/feed"},
                ]
            }
        }
        
        # 기술 키워드 정의 (문서 기반으로 확장)
        self.tech_keywords = {
            "하드웨어 혁신": [
                "spintronics", "스핀트로닉스", "magnonics", "마그노닉스", 
                "quantum computing", "양자컴퓨팅", "semiconductor", "반도체",
                "chip", "칩", "processor", "프로세서", "MRAM", "spin wave"
            ],
            "AI/ML": [
                "artificial intelligence", "인공지능", "machine learning", "머신러닝",
                "deep learning", "딥러닝", "neural network", "신경망",
                "GPT", "LLM", "transformer", "bayesian", "베이지안"
            ],
            "보안/해킹": [
                "cybersecurity", "사이버보안", "hacking", "해킹", "data security", "데이터보안",
                "privacy", "프라이버시", "encryption", "암호화", "adversarial attack",
                "data poisoning", "model stealing"
            ],
            "법률/규제": [
                "AI law", "AI법", "regulation", "규제", "GDPR", "data protection",
                "governance", "거버넌스", "policy", "정책", "compliance", "준수",
                "ethics", "윤리", "liability", "책임"
            ],
            "자율시스템": [
                "autonomous", "자율", "self-driving", "자율주행", "robot", "로봇",
                "drone", "드론", "IoT", "smart city", "스마트시티"
            ]
        }
        
        # 국가 감지 키워드
        self.country_keywords = {
            "미국": ["USA", "United States", "America", "US", "American"],
            "중국": ["China", "Chinese", "Beijing", "Shanghai", "중국"],
            "일본": ["Japan", "Japanese", "Tokyo", "일본"],
            "독일": ["Germany", "German", "Berlin", "독일"],
            "영국": ["UK", "Britain", "British", "London", "영국"],
            "프랑스": ["France", "French", "Paris", "프랑스"],
            "한국": ["Korea", "Korean", "Seoul", "South Korea", "한국"],
            "이스라엘": ["Israel", "Israeli", "Tel Aviv", "이스라엘"],
            "싱가포르": ["Singapore", "싱가포르"],
            "캐나다": ["Canada", "Canadian", "Toronto", "캐나다"]
        }

    def detect_country(self, text: str) -> str:
        """텍스트에서 국가 감지"""
        text_lower = text.lower()
        
        country_scores = {}
        for country, keywords in self.country_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            if score > 0:
                country_scores[country] = score
        
        if country_scores:
            return max(country_scores, key=country_scores.get)
        return "기타"

    def categorize_article(self, title: str, content: str) -> str:
        """기사 카테고리 분류"""
        text = f"{title} {content}".lower()
        
        category_scores = {}
        for category, keywords in self.tech_keywords.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            category_scores[category] = score
        
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return "기타"

    def extract_from_rss(self, rss_url: str, source_name: str) -> List[Dict]:
        """RSS 피드에서 기사 추출"""
        articles = []
        
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:10]:  # 최신 10개 기사만
                # 기본 정보 추출
                title = entry.get('title', '')
                link = entry.get('link', '')
                description = entry.get('description', '')
                published = entry.get('published_parsed', None)
                
                # 날짜 처리
                if published:
                    pub_date = datetime(*published[:6]).strftime('%Y-%m-%d')
                else:
                    pub_date = datetime.now().strftime('%Y-%m-%d')
                
                # 본문 내용 가져오기 (요약에서 HTML 태그 제거)
                content = BeautifulSoup(description, 'html.parser').get_text()
                
                # 전체 기사 내용 가져오기 시도
                try:
                    full_content = self.get_article_content(link)
                    if full_content:
                        content = full_content[:1000]  # 처음 1000자만
                except:
                    pass
                
                # 국가 및 카테고리 분류
                country = self.detect_country(f"{title} {content}")
                category = self.categorize_article(title, content)
                
                # 감정 분석
                sentiment = self.analyze_sentiment(content)
                
                article = {
                    'title': title,
                    'content': content,
                    'url': link,
                    'source': source_name,
                    'country': country,
                    'category': category,
                    'date': pub_date,
                    'sentiment': sentiment,
                    'scraped_at': datetime.now().isoformat()
                }
                
                articles.append(article)
                
        except Exception as e:
            logger.error(f"RSS 추출 오류 {rss_url}: {e}")
        
        return articles

    def get_article_content(self, url: str) -> Optional[str]:
        """개별 기사 페이지에서 본문 추출"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 일반적인 기사 본문 선택자들
            content_selectors = [
                'article', '.article-content', '.post-content',
                '.entry-content', '.story-body', '.article-body',
                'main', '.main-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # 스크립트와 스타일 태그 제거
                    for script in content_elem(["script", "style", "nav", "header", "footer"]):
                        script.decompose()
                    
                    text = content_elem.get_text()
                    # 텍스트 정리
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 200:  # 충분한 내용이 있는 경우만
                        return text
            
            return None
            
        except Exception as e:
            logger.warning(f"기사 내용 추출 실패 {url}: {e}")
            return None

    def analyze_sentiment(self, text: str) -> float:
        """감정 분석"""
        try:
            blob = TextBlob(text)
            return round(blob.sentiment.polarity, 3)
        except:
            return 0.0

    def scrape_google_news_search(self, query: str, country_code: str = "us") -> List[Dict]:
        """구글 뉴스 검색 결과 스크래핑"""
        articles = []
        
        try:
            # 구글 뉴스 검색 URL (실제 구현에서는 News API 사용 권장)
            search_url = f"https://news.google.com/rss/search?q={query}&hl=en-{country_code}&gl={country_code.upper()}&ceid={country_code.upper()}:en"
            
            feed = feedparser.parse(search_url)
            
            for entry in feed.entries[:5]:
                title = entry.get('title', '')
                link = entry.get('link', '')
                description = entry.get('description', '')
                published = entry.get('published_parsed', None)
                
                if published:
                    pub_date = datetime(*published[:6]).strftime('%Y-%m-%d')
                else:
                    pub_date = datetime.now().strftime('%Y-%m-%d')
                
                content = BeautifulSoup(description, 'html.parser').get_text()
                country = self.detect_country(f"{title} {content}")
                category = self.categorize_article(title, content)
                sentiment = self.analyze_sentiment(content)
                
                article = {
                    'title': title,
                    'content': content,
                    'url': link,
                    'source': 'Google News',
                    'country': country,
                    'category': category,
                    'date': pub_date,
                    'sentiment': sentiment,
                    'scraped_at': datetime.now().isoformat()
                }
                
                articles.append(article)
                
        except Exception as e:
            logger.error(f"구글 뉴스 검색 오류: {e}")
        
        return articles

    def scrape_all_sources(self) -> List[Dict]:
        """모든 소스에서 기사 수집"""
        all_articles = []
        
        # RSS 피드 수집
        for region, config in self.news_sources.items():
            for source in config['sources']:
                logger.info(f"RSS 수집 중: {source['name']}")
                articles = self.extract_from_rss(source['rss'], source['name'])
                all_articles.extend(articles)
                
                # 요청 간격 조절
                time.sleep(random.uniform(1, 3))
        
        # 기술 키워드별 구글 뉴스 검색
        tech_searches = [
            "spintronics technology", "quantum computing breakthrough",
            "AI regulation law", "cybersecurity attack", "autonomous vehicle"
        ]
        
        for query in tech_searches:
            logger.info(f"구글 뉴스 검색: {query}")
            articles = self.scrape_google_news_search(query)
            all_articles.extend(articles)
            time.sleep(random.uniform(2, 4))
        
        # 중복 제거 (URL 기준)
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            if article['url'] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article['url'])
        
        logger.info(f"총 {len(unique_articles)}개의 고유 기사 수집 완료")
        return unique_articles

    def save_to_json(self, articles: List[Dict], filename: str = "tech_news_data.json"):
        """JSON 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"데이터 저장 완료: {filename}")

    def load_from_json(self, filename: str = "tech_news_data.json") -> List[Dict]:
        """JSON 파일에서 로드"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"파일을 찾을 수 없음: {filename}")
            return []

# 메인 실행부
if __name__ == "__main__":
    scraper = GlobalTechNewsScraper()
    
    # 뉴스 수집
    articles = scraper.scrape_all_sources()
    
    # 데이터 저장
    scraper.save_to_json(articles)
    
    # 결과 요약
    print(f"\n=== 수집 결과 요약 ===")
    print(f"총 기사 수: {len(articles)}")
    
    # 국가별 분포
    countries = {}
    categories = {}
    
    for article in articles:
        countries[article['country']] = countries.get(article['country'], 0) + 1
        categories[article['category']] = categories.get(article['category'], 0) + 1
    
    print(f"\n국가별 분포:")
    for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True):
        print(f"  {country}: {count}개")
    
    print(f"\n카테고리별 분포:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}개")
