# tests/integration/test_full_pipeline.py
import pytest
import json
import tempfile
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from news_scraper import GlobalTechNewsScraper
from advanced_analytics import TechTrendAnalyzer, AdvancedVisualizer
from data_collector import DataCollectionScheduler
from utils import DataValidator, ConfigManager, DataCache

class TestFullPipeline:
    """전체 데이터 파이프라인 통합 테스트"""
    
    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 픽스처"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def sample_articles(self):
        """샘플 기사 데이터"""
        base_date = datetime.now() - timedelta(days=10)
        
        articles = []
        countries = ["미국", "중국", "일본", "독일", "한국"]
        categories = ["AI/ML", "하드웨어 혁신", "보안/해킹", "법률/규제", "자율시스템"]
        
        for i in range(50):
            article = {
                'title': f'기술 뉴스 제목 {i+1}',
                'content': f'이것은 {categories[i % len(categories)]} 분야의 기사 내용입니다. 중요한 기술 발전에 대한 내용을 담고 있습니다.',
                'country': countries[i % len(countries)],
                'category': categories[i % len(categories)],
                'date': (base_date + timedelta(days=i % 10)).strftime('%Y-%m-%d'),
                'sentiment': (i % 5 - 2) * 0.2,  # -0.4 ~ 0.4 사이의 값
                'source': f'Test Source {i % 3 + 1}',
                'url': f'https://example.com/article/{i+1}',
                'scraped_at': datetime.now().isoformat()
            }
            articles.append(article)
        
        return articles
    
    def test_end_to_end_data_flow(self, temp_dir, sample_articles):
        """종단간 데이터 흐름 테스트"""
        # 1. 데이터 수집 시뮬레이션
        scraper = GlobalTechNewsScraper()
        
        # 임시 파일에 샘플 데이터 저장
        data_file = os.path.join(temp_dir, 'test_articles.json')
        scraper.save_to_json(sample_articles, data_file)
        
        # 2. 데이터 로드 및 검증
        loaded_articles = scraper.load_from_json(data_file)
        assert len(loaded_articles) == len(sample_articles)
        
        # 3. 데이터 유효성 검증
        validator = DataValidator()
        validation_result = validator.validate_articles_batch(loaded_articles)
        
        assert validation_result['total_articles'] == len(sample_articles)
        assert validation_result['validation_rate'] >= 0.9  # 90% 이상 유효해야 함
        
        # 4. 분석 실행
        analyzer = TechTrendAnalyzer(loaded_articles)
        
        # 기본 분석 테스트
        insights = analyzer.generate_insights_report()
        assert len(insights) > 0
        assert '기사 수' in insights
        
        # 트렌드 분석 테스트
        trending = analyzer.detect_emerging_trends()
        assert isinstance(trending, dict)
        
        # 국가별 분석 테스트
        country_focus = analyzer.analyze_country_tech_focus()
        assert isinstance(country_focus, dict)
        assert len(country_focus) > 0
        
        # 5. 시각화 생성 테스트
        visualizer = AdvancedVisualizer(analyzer)
        
        # 네트워크 시각화 테스트
        network_fig = visualizer.create_network_visualization()
        assert network_fig is not None
        
        # 감정 타임라인 테스트
        sentiment_fig = visualizer.create_sentiment_timeline()
        assert sentiment_fig is not None
        
        print("✅ 종단간 데이터 흐름 테스트 통과")
    
    def test_data_collection_scheduler_integration(self, temp_dir, sample_articles):
        """데이터 수집 스케줄러 통합 테스트"""
        # 임시 환경 설정
        os.environ['DATA_DIR'] = temp_dir
        
        # 스케줄러 초기화
        scheduler = DataCollectionScheduler()
        
        # Mock 데이터로 수집 시뮬레이션
        with patch.object(scheduler.scraper, 'scrape_all_sources', return_value=sample_articles):
            success = scheduler.collect_news_data()
            assert success
        
        # 수집 통계 확인
        stats = scheduler.collection_stats
        assert stats['total_collections'] > 0
        assert stats['successful_collections'] > 0
        
        # 상태 확인
        health_status = scheduler.get_health_status()
        assert health_status['status'] in ['healthy', 'warning']
        
        print("✅ 데이터 수집 스케줄러 통합 테스트 통과")
    
    def test_config_and_cache_integration(self, temp_dir):
        """설정 관리 및 캐시 통합 테스트"""
        # 임시 설정 파일 경로
        config_file = os.path.join(temp_dir, 'test_config.json')
        cache_dir = os.path.join(temp_dir, 'cache')
        
        # 설정 관리자 테스트
        config = ConfigManager(config_file)
        
        # 기본 설정 확인
        default_countries = config.get('ui.default_countries')
        assert isinstance(default_countries, list)
        
        # 설정 변경 및 저장
        config.set('test.value', 'integration_test')
        assert config.get('test.value') == 'integration_test'
        
        # 캐시 시스템 테스트
        cache = DataCache(cache_dir)
        
        test_data = {'key': 'value', 'timestamp': datetime.now().isoformat()}
        cache.set('test_key', test_data)
        
        cached_data = cache.get('test_key')
        assert cached_data == test_data
        
        # 캐시 만료 테스트
        expired_data = cache.get('test_key', max_age_hours=0)  # 즉시 만료
        assert expired_data is None
        
        print("✅ 설정 관리 및 캐시 통합 테스트 통과")
    
    def test_error_handling_and_recovery(self, temp_dir, sample_articles):
        """에러 처리 및 복구 테스트"""
        # 잘못된 데이터로 테스트
        invalid_articles = [
            {
                'title': '',  # 빈 제목
                'content': 'Test content',
                'country': 'InvalidCountry',
                'category': 'InvalidCategory',
                'date': '2025-13-45',  # 잘못된 날짜
                'sentiment': 'invalid',  # 잘못된 감정 점수
            },
            {
                'title': 'Valid Title',
                'content': 'Valid content',
                'country': '미국',
                'category': 'AI/ML',
                'date': '2025-07-05',
                'sentiment': 0.5,
                'source': 'Test Source',
                'url': 'https://example.com/valid'
            }
        ]
        
        # 데이터 검증
        validator = DataValidator()
        validation_result = validator.validate_articles_batch(invalid_articles)
        
        # 일부 기사는 유효하지 않을 것
        assert validation_result['invalid_articles'] > 0
        assert validation_result['valid_articles'] > 0
        
        # 유효한 데이터만으로 분석 진행 가능한지 확인
        valid_articles = [article for article in invalid_articles 
                         if validator.validate_article(article)[0]]
        
        if valid_articles:
            analyzer = TechTrendAnalyzer(valid_articles)
            insights = analyzer.generate_insights_report()
            assert len(insights) > 0
        
        print("✅ 에러 처리 및 복구 테스트 통과")
    
    def test_performance_with_large_dataset(self, temp_dir):
        """대용량 데이터셋 성능 테스트"""
        # 대용량 샘플 데이터 생성 (1000개 기사)
        large_dataset = []
        countries = ["미국", "중국", "일본", "독일", "한국"]
        categories = ["AI/ML", "하드웨어 혁신", "보안/해킹", "법률/규제", "자율시스템"]
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(1000):
            article = {
                'title': f'Performance Test Article {i+1}',
                'content': f'This is article content for performance testing. Category: {categories[i % len(categories)]}. ' * 10,
                'country': countries[i % len(countries)],
                'category': categories[i % len(categories)],
                'date': (base_date + timedelta(days=i % 30)).strftime('%Y-%m-%d'),
                'sentiment': ((i % 10) - 5) * 0.1,
                'source': f'Performance Source {i % 5 + 1}',
                'url': f'https://example.com/perf/{i+1}',
                'scraped_at': datetime.now().isoformat()
            }
            large_dataset.append(article)
        
        # 성능 측정 시작
        start_time = datetime.now()
        
        # 분석 실행
        analyzer = TechTrendAnalyzer(large_dataset)
        
        # 기본 분석
        insights = analyzer.generate_insights_report()
        trending = analyzer.detect_emerging_trends()
        country_focus = analyzer.analyze_country_tech_focus()
        
        # 시각화 생성
        visualizer = AdvancedVisualizer(analyzer)
        network_fig = visualizer.create_network_visualization()
        sentiment_fig = visualizer.create_sentiment_timeline()
        
        # 성능 측정 종료
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # 성능 기준: 1000개 기사 처리가 30초 이내
        assert processing_time < 30, f"처리 시간이 너무 깁니다: {processing_time}초"
        
        # 결과 검증
        assert len(insights) > 0
        assert isinstance(trending, dict)
        assert isinstance(country_focus, dict)
        assert network_fig is not None
        assert sentiment_fig is not None
        
        print(f"✅ 대용량 데이터셋 성능 테스트 통과 (처리 시간: {processing_time:.2f}초)")
    
    def test_concurrent_operations(self, temp_dir, sample_articles):
        """동시 작업 테스트"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker_analysis(articles, worker_id):
            """워커 스레드에서 실행할 분석 작업"""
            try:
                analyzer = TechTrendAnalyzer(articles)
                insights = analyzer.generate_insights_report()
                results.put(f"Worker {worker_id}: {len(insights)} characters")
            except Exception as e:
                errors.put(f"Worker {worker_id}: {str(e)}")
        
        # 5개의 워커 스레드로 동시 분석 실행
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=worker_analysis, 
                args=(sample_articles, i+1)
            )
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join(timeout=30)  # 30초 타임아웃
        
        # 결과 확인
        assert errors.empty(), f"에러 발생: {list(errors.queue)}"
        assert results.qsize() == 5, f"예상 결과 수와 다름: {results.qsize()}"
        
        print("✅ 동시 작업 테스트 통과")
    
    def test_data_consistency_across_restarts(self, temp_dir, sample_articles):
        """재시작 간 데이터 일관성 테스트"""
        data_file = os.path.join(temp_dir, 'consistency_test.json')
        
        # 첫 번째 실행: 데이터 저장 및 분석
        scraper1 = GlobalTechNewsScraper()
        scraper1.save_to_json(sample_articles, data_file)
        
        analyzer1 = TechTrendAnalyzer(sample_articles)
        insights1 = analyzer1.generate_insights_report()
        country_focus1 = analyzer1.analyze_country_tech_focus()
        
        # 두 번째 실행: 데이터 로드 및 재분석
        scraper2 = GlobalTechNewsScraper()
        loaded_articles = scraper2.load_from_json(data_file)
        
        analyzer2 = TechTrendAnalyzer(loaded_articles)
        insights2 = analyzer2.generate_insights_report()
        country_focus2 = analyzer2.analyze_country_tech_focus()
        
        # 일관성 확인
        assert len(loaded_articles) == len(sample_articles)
        
        # 분석 결과 일관성 (대략적인 비교)
        assert len(insights1) == len(insights2)
        assert len(country_focus1) == len(country_focus2)
        
        # 각 국가의 기술 집중도 비교
        for country in country_focus1:
            if country in country_focus2:
                focus1 = country_focus1[country]
                focus2 = country_focus2[country]
                # 키가 동일한지 확인
                assert set(focus1.keys()) == set(focus2.keys())
        
        print("✅ 재시작 간 데이터 일관성 테스트 통과")

class TestAPIIntegration:
    """API 통합 테스트 (향후 확장용)"""
    
    @pytest.mark.skip(reason="API 엔드포인트가 아직 구현되지 않음")
    def test_rest_api_endpoints(self):
        """REST API 엔드포인트 테스트"""
        # 향후 API 개발 시 활성화
        pass
    
    @pytest.mark.skip(reason="실시간 업데이트 기능이 아직 구현되지 않음")
    def test_websocket_real_time_updates(self):
        """WebSocket 실시간 업데이트 테스트"""
        # 향후 실시간 기능 개발 시 활성화
        pass

class TestSecurityIntegration:
    """보안 통합 테스트"""
    
    def test_input_sanitization(self, sample_articles):
        """입력 데이터 검증 및 새니타이제이션 테스트"""
        # 악성 입력 시뮬레이션
        malicious_articles = [
            {
                'title': '<script>alert("XSS")</script>',
                'content': 'SELECT * FROM users; DROP TABLE articles;',
                'country': '미국',
                'category': 'AI/ML',
                'date': '2025-07-05',
                'sentiment': 0.5,
                'source': 'Test Source',
                'url': 'javascript:alert("XSS")'
            }
        ]
        
        # 검증 통과하는지 확인
        validator = DataValidator()
        is_valid, errors = validator.validate_article(malicious_articles[0])
        
        # 기본 유효성은 통과하되, 실제 사용 시 적절히 새니타이즈되어야 함
        if is_valid:
            analyzer = TechTrendAnalyzer(malicious_articles)
            insights = analyzer.generate_insights_report()
            
            # 스크립트 태그가 그대로 노출되지 않는지 확인
            assert '<script>' not in insights
            assert 'DROP TABLE' not in insights
        
        print("✅ 입력 데이터 보안 테스트 통과")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
