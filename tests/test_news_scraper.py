
# tests/test_news_scraper.py
'''import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import feedparser
import requests

# 테스트 대상 모듈 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_scraper import GlobalTechNewsScraper

class TestGlobalTechNewsScraper:
    """뉴스 스크래퍼 테스트 클래스"""
    
    @pytest.fixture
    def scraper(self):
        """스크래퍼 인스턴스 픽스처"""
        return GlobalTechNewsScraper()
    
    @pytest.fixture
    def sample_article_data(self):
        """샘플 기사 데이터"""
        return {
            'title': '스핀트로닉스 기술의 새로운 돌파구',
            'content': '연구진이 스핀트로닉스 기술을 이용해 기존보다 100배 빠른 연산속도를 달성했다.',
            'url': 'https://example.com/article/1',
            'source': 'Tech Review',
            'published': '2025-07-05',
        }
    
    @pytest.fixture
    def mock_rss_feed(self):
        """Mock RSS 피드 데이터"""
        return {
            'entries': [
                {
                    'title': 'AI breakthrough in quantum computing',
                    'link': 'https://example.com/ai-quantum',
                    'description': 'Scientists achieve new AI breakthrough in quantum computing research.',
                    'published_parsed': (2025, 7, 5, 10, 30, 0, 0, 0, 0)
                },
                {
                    'title': 'Cybersecurity threat detected in IoT devices',
                    'link': 'https://example.com/iot-security',
                    'description': 'New cybersecurity vulnerabilities found in smart home devices.',
                    'published_parsed': (2025, 7, 4, 14, 15, 0, 0, 0, 0)
                }
            ]
        }

    def test_detect_country_valid_keywords(self, scraper):
        """국가 감지 테스트 - 유효한 키워드"""
        # 미국 키워드 테스트
        text = "The United States government announced new AI regulations."
        assert scraper.detect_country(text) == "미국"
        
        # 중국 키워드 테스트
        text = "Beijing-based company develops quantum computer."
        assert scraper.detect_country(text) == "중국"
        
        # 한국 키워드 테스트
        text = "South Korean tech giant releases new semiconductor."
        assert scraper.detect_country(text) == "한국"
    
    def test_detect_country_no_keywords(self, scraper):
        """국가 감지 테스트 - 키워드 없음"""
        text = "Generic technology news without country mentions."
        assert scraper.detect_country(text) == "기타"
    
    def test_categorize_article_ai_ml(self, scraper):
        """기사 카테고리 분류 테스트 - AI/ML"""
        title = "New machine learning algorithm improves accuracy"
        content = "Researchers developed a deep learning model for better AI performance."
        
        category = scraper.categorize_article(title, content)
        assert category == "AI/ML"
    
    def test_categorize_article_hardware(self, scraper):
        """기사 카테고리 분류 테스트 - 하드웨어"""
        title = "Breakthrough in spintronics technology"
        content = "Scientists achieve quantum computing milestone with new semiconductor design."
        
        category = scraper.categorize_article(title, content)
        assert category == "하드웨어 혁신"
    
    def test_categorize_article_security(self, scraper):
        """기사 카테고리 분류 테스트 - 보안"""
        title = "Major cybersecurity breach detected"
        content = "Hackers exploited data security vulnerabilities in enterprise systems."
        
        category = scraper.categorize_article(title, content)
        assert category == "보안/해킹"
    
    def test_categorize_article_no_match(self, scraper):
        """기사 카테고리 분류 테스트 - 매칭 없음"""
        title = "Weather forecast for tomorrow"
        content = "Sunny skies expected with mild temperatures."
        
        category = scraper.categorize_article(title, content)
        assert category == "기타"
    
    @patch('feedparser.parse')
    def test_extract_from_rss_success(self, mock_parse, scraper, mock_rss_feed):
        """RSS 추출 성공 테스트"""
        mock_parse.return_value = mock_rss_feed
        
        articles = scraper.extract_from_rss('https://example.com/rss', 'Test Source')
        
        assert len(articles) == 2
        assert articles[0]['title'] == 'AI breakthrough in quantum computing'
        assert articles[0]['source'] == 'Test Source'
        assert articles[0]['country'] in scraper.country_keywords.keys() or articles[0]['country'] == '기타'
        assert articles[0]['category'] in scraper.tech_keywords.keys() or articles[0]['category'] == '기타'
    
    @patch('feedparser.parse')
    def test_extract_from_rss_failure(self, mock_parse, scraper):
        """RSS 추출 실패 테스트"""
        mock_parse.side_effect = Exception("Network error")
        
        articles = scraper.extract_from_rss('https://invalid-url.com/rss', 'Test Source')
        
        assert articles == []
    
    def test_analyze_sentiment_positive(self, scraper):
        """감정 분석 테스트 - 긍정적"""
        text = "Amazing breakthrough! Scientists achieve incredible success in technology."
        sentiment = scraper.analyze_sentiment(text)
        
        assert sentiment > 0
        assert -1 <= sentiment <= 1
    
    def test_analyze_sentiment_negative(self, scraper):
        """감정 분석 테스트 - 부정적"""
        text = "Terrible security breach causes massive damage and failure."
        sentiment = scraper.analyze_sentiment(text)
        
        assert sentiment < 0
        assert -1 <= sentiment <= 1
    
    def test_analyze_sentiment_neutral(self, scraper):
        """감정 분석 테스트 - 중립적"""
        text = "The report contains statistical data and technical specifications."
        sentiment = scraper.analyze_sentiment(text)
        
        assert -0.1 <= sentiment <= 0.1
    
    @patch('requests.Session.get')
    def test_get_article_content_success(self, mock_get, scraper):
        """기사 내용 추출 성공 테스트"""
        mock_response = Mock()
        mock_response.content = '''
        <html>
            <body>
                <article>
                    <h1>Test Article</h1>
                    <p>This is the main content of the article.</p>
                    <p>Additional paragraph with important information.</p>
                </article>
            </body>
        </html>
        '''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        content = scraper.get_article_content('https://example.com/article')
        
        assert content is not None
        assert 'Test Article' in content
        assert 'main content' in content
    
    @patch('requests.Session.get')
    def test_get_article_content_failure(self, mock_get, scraper):
        """기사 내용 추출 실패 테스트"""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        content = scraper.get_article_content('https://invalid-url.com/article')
        
        assert content is None
    
    @patch('feedparser.parse')
    def test_scrape_google_news_search(self, mock_parse, scraper):
        """구글 뉴스 검색 테스트"""
        mock_feed = {
            'entries': [
                {
                    'title': 'AI technology advances',
                    'link': 'https://news.google.com/article1',
                    'description': 'Artificial intelligence makes significant progress.',
                    'published_parsed': (2025, 7, 5, 12, 0, 0, 0, 0, 0)
                }
            ]
        }
        mock_parse.return_value = mock_feed
        
        articles = scraper.scrape_google_news_search('artificial intelligence')
        
        assert len(articles) == 1
        assert articles[0]['title'] == 'AI technology advances'
        assert articles[0]['source'] == 'Google News'
    
    def test_save_and_load_json(self, scraper, sample_article_data, tmp_path):
        """JSON 저장/로드 테스트"""
        test_file = tmp_path / "test_articles.json"
        articles = [sample_article_data]
        
        # 저장 테스트
        scraper.save_to_json(articles, str(test_file))
        assert test_file.exists()
        
        # 로드 테스트
        loaded_articles = scraper.load_from_json(str(test_file))
        assert len(loaded_articles) == 1
        assert loaded_articles[0]['title'] == sample_article_data['title']
    
    def test_load_json_file_not_found(self, scraper):
        """존재하지 않는 JSON 파일 로드 테스트"""
        articles = scraper.load_from_json('nonexistent_file.json')
        assert articles == []
    
    @pytest.mark.integration
    @patch('feedparser.parse')
    @patch('requests.Session.get')
    def test_scrape_all_sources_integration(self, mock_get, mock_parse, scraper):
        """전체 소스 스크래핑 통합 테스트"""
        # RSS 피드 목킹
        mock_parse.return_value = {
            'entries': [
                {
                    'title': 'Tech news article',
                    'link': 'https://example.com/tech-news',
                    'description': 'Technology advancement in AI field.',
                    'published_parsed': (2025, 7, 5, 10, 0, 0, 0, 0, 0)
                }
            ]
        }
        
        # HTTP 요청 목킹
        mock_response = Mock()
        mock_response.content = '<article><p>Article content</p></article>'
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        articles = scraper.scrape_all_sources()
        
        # 기본적인 검증
        assert isinstance(articles, list)
        
        if articles:  # 기사가 수집된 경우
            article = articles[0]
            assert 'title' in article
            assert 'content' in article
            assert 'country' in article
            assert 'category' in article
            assert 'date' in article
            assert 'sentiment' in article

    def test_tech_keywords_coverage(self, scraper):
        """기술 키워드 커버리지 테스트"""
        # 모든 기술 카테고리에 키워드가 있는지 확인
        for category, keywords in scraper.tech_keywords.items():
            assert len(keywords) > 0, f"카테고리 '{category}'에 키워드가 없습니다."
            
            # 각 키워드가 문자열인지 확인
            for keyword in keywords:
                assert isinstance(keyword, str), f"키워드 '{keyword}'가 문자열이 아닙니다."
                assert len(keyword.strip()) > 0, f"빈 키워드가 발견되었습니다."
    
    def test_country_keywords_coverage(self, scraper):
        """국가 키워드 커버리지 테스트"""
        # 모든 국가에 키워드가 있는지 확인
        for country, keywords in scraper.country_keywords.items():
            assert len(keywords) > 0, f"국가 '{country}'에 키워드가 없습니다."
            
            # 각 키워드가 문자열인지 확인
            for keyword in keywords:
                assert isinstance(keyword, str), f"키워드 '{keyword}'가 문자열이 아닙니다."
                assert len(keyword.strip()) > 0, f"빈 키워드가 발견되었습니다."

class TestScraperPerformance:
    """스크래퍼 성능 테스트"""
    
    @pytest.fixture
    def scraper(self):
        return GlobalTechNewsScraper()
    
    def test_categorize_article_performance(self, scraper, benchmark):
        """기사 분류 성능 테스트"""
        title = "AI breakthrough in quantum computing technology"
        content = "Researchers have achieved a significant breakthrough in artificial intelligence and quantum computing, developing new machine learning algorithms that leverage quantum mechanics for enhanced computational capabilities."
        
        result = benchmark(scraper.categorize_article, title, content)
        assert result in scraper.tech_keywords.keys() or result == "기타"
    
    def test_sentiment_analysis_performance(self, scraper, benchmark):
        """감정 분석 성능 테스트"""
        text = "This is an amazing technological breakthrough that will revolutionize the industry and bring significant benefits to society."
        
        result = benchmark(scraper.analyze_sentiment, text)
        assert -1 <= result <= 1

# 파라미터화된 테스트
class TestParameterizedScenarios:
    """파라미터화된 시나리오 테스트"""
    
    @pytest.fixture
    def scraper(self):
        return GlobalTechNewsScraper()
    
    @pytest.mark.parametrize("text,expected_country", [
        ("US government announces new policy", "미국"),
        ("Chinese company develops AI", "중국"),
        ("Japanese researchers achieve breakthrough", "일본"),
        ("German automotive industry", "독일"),
        ("UK Brexit negotiations", "영국"),
        ("French technology sector", "프랑스"),
        ("South Korean electronics", "한국"),
        ("Israeli cybersecurity firm", "이스라엘"),
        ("Singapore fintech startup", "싱가포르"),
        ("Canadian AI research", "캐나다"),
        ("Random technology news", "기타"),
    ])
    def test_country_detection_scenarios(self, scraper, text, expected_country):
        """다양한 국가 감지 시나리오 테스트"""
        result = scraper.detect_country(text)
        assert result == expected_country
    
    @pytest.mark.parametrize("title,content,expected_category", [
        ("AI revolution", "machine learning deep learning", "AI/ML"),
        ("Quantum chip", "spintronics semiconductor quantum computing", "하드웨어 혁신"),
        ("Data breach", "cybersecurity hacking data security", "보안/해킹"),
        ("New AI law", "regulation policy governance AI법", "법률/규제"),
        ("Self-driving car", "autonomous vehicle robot drone", "자율시스템"),
        ("Weather report", "sunny cloudy temperature", "기타"),
    ])
    def test_categorization_scenarios(self, scraper, title, content, expected_category):
        """다양한 카테고리 분류 시나리오 테스트"""
        result = scraper.categorize_article(title, content)
        assert result == expected_category

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])'''
