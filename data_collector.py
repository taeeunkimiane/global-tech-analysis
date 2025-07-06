# data_collector.py - 정기 데이터 수집 및 업데이트
import os
import json
import time
import schedule
from datetime import datetime, timedelta
from typing import List, Dict
import logging
from pathlib import Path

# 커스텀 모듈 import
from news_scraper import GlobalTechNewsScraper
from advanced_analytics import TechTrendAnalyzer
from utils import (
    setup_logging, 
    DataCache, 
    ConfigManager, 
    DataValidator,
    PerformanceMonitor,
    ErrorHandler
)

class DataCollectionScheduler:
    """데이터 수집 스케줄러"""
    
    def __init__(self):
        # 로깅 설정
        self.logger = setup_logging(
            level="INFO", 
            log_file="logs/data_collector.log"
        )
        
        # 유틸리티 초기화
        self.cache = DataCache()
        self.config = ConfigManager()
        self.validator = DataValidator()
        self.monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        
        # 스크래퍼 초기화
        self.scraper = GlobalTechNewsScraper()
        
        # 데이터 파일 경로
        self.data_file = Path("data/tech_news_data.json")
        self.stats_file = Path("data/collection_stats.json")
        
        # 수집 통계
        self.collection_stats = self._load_stats()
        
        self.logger.info("데이터 수집 스케줄러 초기화 완료")
    
    def _load_stats(self) -> Dict:
        """수집 통계 로드"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"통계 파일 로드 실패: {e}")
        
        return {
            'total_collections': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'total_articles_collected': 0,
            'last_collection_time': None,
            'last_successful_collection': None,
            'collection_history': []
        }
    
    def _save_stats(self):
        """수집 통계 저장"""
        try:
            self.stats_file.parent.mkdir(exist_ok=True)
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.collection_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"통계 파일 저장 실패: {e}")
    
    @PerformanceMonitor().timing_decorator("collect_news_data")
    def collect_news_data(self) -> bool:
        """뉴스 데이터 수집 실행"""
        collection_start = datetime.now()
        self.logger.info("=== 뉴스 데이터 수집 시작 ===")
        
        try:
            # 기존 데이터 로드
            existing_articles = self._load_existing_data()
            existing_urls = {article.get('url', '') for article in existing_articles}
            
            # 새 데이터 수집
            self.logger.info("새 기사 수집 중...")
            new_articles = self.scraper.scrape_all_sources()
            
            if not new_articles:
                self.logger.warning("수집된 새 기사가 없습니다.")
                self._update_stats(collection_start, False, 0)
                return False
            
            # 중복 제거 및 데이터 검증
            self.logger.info("데이터 검증 및 중복 제거 중...")
            validated_articles = []
            new_article_count = 0
            
            for article in new_articles:
                # 중복 확인
                if article.get('url') in existing_urls:
                    continue
                
                # 데이터 검증
                is_valid, errors = self.validator.validate_article(article)
                if is_valid:
                    validated_articles.append(article)
                    new_article_count += 1
                else:
                    self.logger.warning(f"유효하지 않은 기사: {errors}")
            
            if not validated_articles:
                self.logger.info("새로 추가할 유효한 기사가 없습니다.")
                self._update_stats(collection_start, True, 0)
                return True
            
            # 기존 데이터와 병합
            all_articles = existing_articles + validated_articles
            
            # 오래된 데이터 정리 (30일 이상 된 기사 제거)
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            filtered_articles = [
                article for article in all_articles
                if article.get('date', '1900-01-01') >= cutoff_date
            ]
            
            # 데이터 저장
            self.logger.info(f"총 {len(filtered_articles)}개 기사 저장 중...")
            self._save_data(filtered_articles)
            
            # 캐시 업데이트
            self.cache.set('latest_articles', filtered_articles[:100])  # 최신 100개만 캐시
            
            # 분석 실행
            self._run_analysis(filtered_articles)
            
            # 통계 업데이트
            self._update_stats(collection_start, True, new_article_count)
            
            self.logger.info(f"=== 수집 완료: 새 기사 {new_article_count}개 추가 ===")
            return True
            
        except Exception as e:
            error_msg = self.error_handler.handle_data_error(e, "데이터 수집")
            self.logger.error(f"데이터 수집 실패: {error_msg}")
            self._update_stats(collection_start, False, 0)
            return False
    
    def _load_existing_data(self) -> List[Dict]:
        """기존 데이터 로드"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"기존 데이터 로드 실패: {e}")
        
        return []
    
    def _save_data(self, articles: List[Dict]):
        """데이터 저장"""
        self.data_file.parent.mkdir(exist_ok=True)
        
        # 백업 생성
        if self.data_file.exists():
            backup_file = self.data_file.with_suffix('.backup.json')
            self.data_file.rename(backup_file)
        
        # 새 데이터 저장
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
    
    def _run_analysis(self, articles: List[Dict]):
        """분석 실행 및 결과 저장"""
        try:
            self.logger.info("데이터 분석 중...")
            
            analyzer = TechTrendAnalyzer(articles)
            
            # 인사이트 리포트 생성
            insights = analyzer.generate_insights_report()
            
            # 신흥 트렌드 분석
            trending_keywords = analyzer.detect_emerging_trends()
            
            # 국가별 기술 집중도
            country_focus = analyzer.analyze_country_tech_focus()
            
            # 분석 결과 저장
            analysis_results = {
                'generated_at': datetime.now().isoformat(),
                'insights_report': insights,
                'trending_keywords': trending_keywords,
                'country_tech_focus': country_focus,
                'total_articles_analyzed': len(articles)
            }
            
            analysis_file = Path("data/latest_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info("분석 완료 및 결과 저장")
            
        except Exception as e:
            self.logger.error(f"분석 실행 실패: {e}")
    
    def _update_stats(self, start_time: datetime, success: bool, new_articles: int):
        """수집 통계 업데이트"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.collection_stats['total_collections'] += 1
        
        if success:
            self.collection_stats['successful_collections'] += 1
            self.collection_stats['total_articles_collected'] += new_articles
            self.collection_stats['last_successful_collection'] = end_time.isoformat()
        else:
            self.collection_stats['failed_collections'] += 1
        
        self.collection_stats['last_collection_time'] = end_time.isoformat()
        
        # 수집 히스토리 추가 (최근 100회만 유지)
        history_entry = {
            'timestamp': end_time.isoformat(),
            'success': success,
            'new_articles': new_articles,
            'duration_seconds': duration
        }
        
        self.collection_stats['collection_history'].append(history_entry)
        
        # 히스토리 크기 제한
        if len(self.collection_stats['collection_history']) > 100:
            self.collection_stats['collection_history'] = \
                self.collection_stats['collection_history'][-100:]
        
        self._save_stats()
    
    def cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            self.logger.info("오래된 데이터 정리 중...")
            
            # 오래된 캐시 파일 정리
            deleted_cache = self.cache.clear("*")
            self.logger.info(f"{deleted_cache}개 캐시 파일 삭제")
            
            # 오래된 로그 파일 정리 (7일 이상)
            log_dir = Path("logs")
            if log_dir.exists():
                cutoff_time = datetime.now() - timedelta(days=7)
                
                for log_file in log_dir.glob("*.log.*"):  # 로테이션된 로그 파일들
                    if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_time:
                        log_file.unlink()
                        self.logger.info(f"오래된 로그 파일 삭제: {log_file}")
            
            self.logger.info("데이터 정리 완료")
            
        except Exception as e:
            self.logger.error(f"데이터 정리 실패: {e}")
    
    def get_health_status(self) -> Dict:
        """시스템 상태 확인"""
        stats = self.collection_stats
        
        # 성공률 계산
        success_rate = 0
        if stats['total_collections'] > 0:
            success_rate = stats['successful_collections'] / stats['total_collections']
        
        # 마지막 성공적인 수집 이후 경과 시간
        last_success = stats.get('last_successful_collection')
        hours_since_success = None
        
        if last_success:
            last_success_time = datetime.fromisoformat(last_success)
            hours_since_success = (datetime.now() - last_success_time).total_seconds() / 3600
        
        # 상태 판정
        status = "healthy"
        if hours_since_success and hours_since_success > 24:
            status = "warning"
        if hours_since_success and hours_since_success > 48:
            status = "critical"
        
        return {
            'status': status,
            'total_collections': stats['total_collections'],
            'success_rate': success_rate,
            'total_articles': stats['total_articles_collected'],
            'hours_since_last_success': hours_since_success,
            'last_collection': stats.get('last_collection_time'),
            'performance_report': self.monitor.get_report()
        }
    
    def run_scheduler(self):
        """스케줄러 실행"""
        # 수집 간격 설정 (환경 변수 또는 설정 파일에서)
        interval_hours = int(os.getenv('COLLECTION_INTERVAL', 21600)) // 3600  # 기본 6시간
        
        # 스케줄 등록
        schedule.every(interval_hours).hours.do(self.collect_news_data)
        schedule.every().day.at("02:00").do(self.cleanup_old_data)  # 매일 새벽 2시 정리
        
        self.logger.info(f"스케줄러 시작 - 수집 간격: {interval_hours}시간")
        
        # 초기 수집 실행
        self.collect_news_data()
        
        # 스케줄 루프
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 확인
                
            except KeyboardInterrupt:
                self.logger.info("스케줄러 중지 요청")
                break
            except Exception as e:
                self.logger.error(f"스케줄러 오류: {e}")
                time.sleep(300)  # 5분 대기 후 재시도

def main():
    """메인 실행 함수"""
    scheduler = DataCollectionScheduler()
    
    # 명령행 인자 처리
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "collect":
            # 단발성 수집
            success = scheduler.collect_news_data()
            sys.exit(0 if success else 1)
            
        elif command == "cleanup":
            # 데이터 정리
            scheduler.cleanup_old_data()
            sys.exit(0)
            
        elif command == "status":
            # 상태 확인
            status = scheduler.get_health_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            sys.exit(0)
            
        elif command == "schedule":
            # 스케줄 모드
            scheduler.run_scheduler()
        
        else:
            print("사용법: python data_collector.py [collect|cleanup|status|schedule]")
            sys.exit(1)
    
    else:
        # 기본값: 스케줄 모드
        scheduler.run_scheduler()

if __name__ == "__main__":
    main()
