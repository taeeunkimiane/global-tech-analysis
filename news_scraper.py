# news_api_integration.py
import requests
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NewsAPIIntegration:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    def get_tech_news(self, query: str = "technology", country: str = "us", page_size: int = 20):
        """News API를 통해 기술 뉴스 가져오기"""
        
        # 1. Everything endpoint (더 많은 소스)
        everything_url = f"{self.base_url}/everything"
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(everything_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                processed_article = {
                    'title': article.get('title', ''),
                    'content': article.get('content', '') or article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'description': article.get('description', ''),
                    'url_to_image': article.get('urlToImage', '')
                }
                articles.append(processed_article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"News API 요청 오류: {e}")
            return []
    
    def get_tech_news_by_sources(self, sources: list):
        """특정 소스들에서 뉴스 가져오기"""
        sources_str = ",".join(sources)
        
        url = f"{self.base_url}/everything"
        params = {
            'sources': sources_str,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 50
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get('articles', [])
        except Exception as e:
            logger.error(f"소스별 뉴스 가져오기 오류: {e}")
            return []

# 사용 예시
if __name__ == "__main__":
    # News API 키 필요 
    API_KEY = "3b0111025078402bafdda5e64844968a"
    
    news_api = NewsAPIIntegration(API_KEY)
    
    # 기술 뉴스 가져오기
    tech_articles = news_api.get_tech_news("artificial intelligence OR quantum computing")
    
    # 특정 소스에서 가져오기
    tech_sources = [
        "techcrunch", "wired", "ars-technica", "the-verge",
        "hacker-news", "techradar", "engadget"
    ]
    
    source_articles = news_api.get_tech_news_by_sources(tech_sources)
    
    print(f"기술 뉴스 {len(tech_articles)}개 수집")
    print(f"소스별 뉴스 {len(source_articles)}개 수집")
