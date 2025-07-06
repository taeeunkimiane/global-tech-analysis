# utils.py - 유틸리티 함수 모음
import os
import json
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import requests
from functools import wraps
import time

# 로깅 설정
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """로깅 시스템 설정"""
    logger = logging.getLogger("GlobalTechAnalysis")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 중복 핸들러 방지
    if logger.handlers:
        logger.handlers.clear()
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택사항)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 캐시 관리
class DataCache:
    """데이터 캐싱 관리 클래스"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = setup_logging()
    
    def _get_cache_key(self, data: Any) -> str:
        """데이터 기반 캐시 키 생성"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # 캐시 만료 시간 확인
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_time > timedelta(hours=max_age_hours):
            cache_file.unlink()  # 만료된 캐시 삭제
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"캐시 로드 실패 {key}: {e}")
            return None
    
    def set(self, key: str, data: Any) -> bool:
        """캐시에 데이터 저장"""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            self.logger.error(f"캐시 저장 실패 {key}: {e}")
            return False
    
    def clear(self, pattern: str = "*") -> int:
        """캐시 파일 삭제"""
        deleted = 0
        for cache_file in self.cache_dir.glob(f"{pattern}.pkl"):
            cache_file.unlink()
            deleted += 1
        
        self.logger.info(f"{deleted}개 캐시 파일 삭제")
        return deleted

# 설정 관리
class ConfigManager:
    """설정 파일 관리 클래스"""
    
    def __init__(self, config_file: str = "config/settings.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)
        self.logger = setup_logging()
        self._config = self._load_config()
    
    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"설정 파일 로드 실패: {e}")
        
        # 기본 설정 반환
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            "news_sources": {
                "update_interval_hours": 6,
                "max_articles_per_source": 50,
                "timeout_seconds": 30
            },
            "analysis": {
                "sentiment_threshold": 0.1,
                "trend_window_days": 7,
                "min_keyword_frequency": 3
            },
            "ui": {
                "default_countries": ["미국", "중국", "일본", "독일", "한국"],
                "default_categories": ["AI/ML", "하드웨어 혁신", "보안/해킹"],
                "chart_color_scheme": "viridis"
            },
            "cache": {
                "enabled": True,
                "ttl_hours": 24,
                "max_size_mb": 500
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회 (점 표기법 지원)"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """설정값 변경"""
        keys = key.split('.')
        config = self._config
        
        # 중첩 딕셔너리 생성
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        return self._save_config()
    
    def _save_config(self) -> bool:
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"설정 파일 저장 실패: {e}")
            return False

# 데이터 검증
class DataValidator:
    """데이터 품질 검증 클래스"""
    
    @staticmethod
    def validate_article(article: Dict) -> Tuple[bool, List[str]]:
        """기사 데이터 유효성 검증"""
        errors = []
        required_fields = ['title', 'content', 'country', 'category', 'date', 'source']
        
        # 필수 필드 확인
        for field in required_fields:
            if field not in article:
                errors.append(f"필수 필드 누락: {field}")
            elif not article[field] or str(article[field]).strip() == "":
                errors.append(f"빈 값: {field}")
        
        # 날짜 형식 확인
        if 'date' in article:
            try:
                datetime.strptime(article['date'], '%Y-%m-%d')
            except ValueError:
                errors.append("날짜 형식 오류 (YYYY-MM-DD 형식이어야 함)")
        
        # 감정 점수 범위 확인
        if 'sentiment' in article:
            try:
                sentiment = float(article['sentiment'])
                if not -1 <= sentiment <= 1:
                    errors.append("감정 점수는 -1~1 범위여야 함")
            except (ValueError, TypeError):
                errors.append("감정 점수는 숫자여야 함")
        
        # 제목/내용 길이 확인
        if 'title' in article and len(article['title']) > 500:
            errors.append("제목이 너무 길음 (500자 이하)")
        
        if 'content' in article and len(article['content']) > 10000:
            errors.append("내용이 너무 길음 (10,000자 이하)")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_articles_batch(articles: List[Dict]) -> Dict:
        """기사 배치 검증"""
        total_count = len(articles)
        valid_count = 0
        error_summary = {}
        
        for article in articles:
            is_valid, errors = DataValidator.validate_article(article)
            
            if is_valid:
                valid_count += 1
            else:
                for error in errors:
                    error_summary[error] = error_summary.get(error, 0) + 1
        
        return {
            'total_articles': total_count,
            'valid_articles': valid_count,
            'invalid_articles': total_count - valid_count,
            'validation_rate': valid_count / total_count if total_count > 0 else 0,
            'error_summary': error_summary
        }

# 성능 모니터링
class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.logger = setup_logging()
    
    def timing_decorator(self, func_name: str = None):
        """함수 실행 시간 측정 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    status = "success"
                except Exception as e:
                    self.logger.error(f"{name} 실행 오류: {e}")
                    status = "error"
                    raise
                finally:
                    execution_time = time.time() - start_time
                    self._record_metric(name, execution_time, status)
                
                return result
            return wrapper
        return decorator
    
    def _record_metric(self, func_name: str, execution_time: float, status: str):
        """메트릭 기록"""
        if func_name not in self.metrics:
            self.metrics[func_name] = {
                'total_calls': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'success_count': 0,
                'error_count': 0
            }
        
        metric = self.metrics[func_name]
        metric['total_calls'] += 1
        metric['total_time'] += execution_time
        metric['avg_time'] = metric['total_time'] / metric['total_calls']
        metric['min_time'] = min(metric['min_time'], execution_time)
        metric['max_time'] = max(metric['max_time'], execution_time)
        
        if status == "success":
            metric['success_count'] += 1
        else:
            metric['error_count'] += 1
    
    def get_report(self) -> str:
        """성능 리포트 생성"""
        if not self.metrics:
            return "성능 데이터가 없습니다."
        
        report = ["=== 성능 모니터링 리포트 ===\n"]
        
        for func_name, metric in self.metrics.items():
            success_rate = (metric['success_count'] / metric['total_calls']) * 100
            
            report.append(f"함수: {func_name}")
            report.append(f"  총 호출 수: {metric['total_calls']}")
            report.append(f"  평균 실행 시간: {metric['avg_time']:.3f}초")
            report.append(f"  최소/최대 시간: {metric['min_time']:.3f}초 / {metric['max_time']:.3f}초")
            report.append(f"  성공률: {success_rate:.1f}%")
            report.append("")
        
        return "\n".join(report)

# 데이터 내보내기
class DataExporter:
    """데이터 내보내기 클래스"""
    
    @staticmethod
    def to_csv(data: List[Dict], filename: str = None) -> str:
        """CSV 형태로 내보내기"""
        df = pd.DataFrame(data)
        
        if filename:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            return filename
        else:
            return df.to_csv(index=False, encoding='utf-8-sig')
    
    @staticmethod
    def to_excel(data: List[Dict], filename: str, sheets: Dict[str, List[Dict]] = None):
        """Excel 형태로 내보내기"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 메인 데이터
            pd.DataFrame(data).to_excel(writer, sheet_name='기사데이터', index=False)
            
            # 추가 시트
            if sheets:
                for sheet_name, sheet_data in sheets.items():
                    pd.DataFrame(sheet_data).to_excel(writer, sheet_name=sheet_name, index=False)
    
    @staticmethod
    def to_json(data: List[Dict], filename: str = None, indent: int = 2) -> str:
        """JSON 형태로 내보내기"""
        json_str = json.dumps(data, ensure_ascii=False, indent=indent)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return filename
        else:
            return json_str

# Streamlit 헬퍼 함수들
class StreamlitUtils:
    """Streamlit 유틸리티 함수들"""
    
    @staticmethod
    def display_metric_cards(metrics: Dict[str, Any], cols: int = 4):
        """메트릭 카드 표시"""
        columns = st.columns(cols)
        
        for i, (title, value) in enumerate(metrics.items()):
            with columns[i % cols]:
                if isinstance(value, dict):
                    st.metric(
                        label=title,
                        value=value.get('value', 'N/A'),
                        delta=value.get('delta', None)
                    )
                else:
                    st.metric(label=title, value=value)
    
    @staticmethod
    def display_data_quality_report(validation_result: Dict):
        """데이터 품질 리포트 표시"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("총 기사 수", validation_result['total_articles'])
        
        with col2:
            st.metric(
                "유효 기사", 
                validation_result['valid_articles'],
                delta=f"{validation_result['validation_rate']:.1%} 유효율"
            )
        
        with col3:
            st.metric("오류 기사", validation_result['invalid_articles'])
        
        # 오류 상세 정보
        if validation_result['error_summary']:
            st.subheader("오류 상세")
            error_df = pd.DataFrame([
                {'오류 유형': error, '발생 횟수': count}
                for error, count in validation_result['error_summary'].items()
            ])
            st.dataframe(error_df)
    
    @staticmethod
    def create_download_button(data: Any, filename: str, label: str = "다운로드"):
        """다운로드 버튼 생성"""
        if filename.endswith('.csv'):
            if isinstance(data, list):
                csv_data = DataExporter.to_csv(data)
            else:
                csv_data = data.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label=label,
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )
        
        elif filename.endswith('.json'):
            if isinstance(data, list):
                json_data = DataExporter.to_json(data)
            else:
                json_data = json.dumps(data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label=label,
                data=json_data,
                file_name=filename,
                mime="application/json"
            )

# 에러 처리
class ErrorHandler:
    """통합 에러 처리 클래스"""
    
    def __init__(self):
        self.logger = setup_logging()
    
    def handle_api_error(self, error: Exception, context: str = "API 호출"):
        """API 에러 처리"""
        if isinstance(error, requests.exceptions.Timeout):
            error_msg = f"{context} 시간 초과"
        elif isinstance(error, requests.exceptions.ConnectionError):
            error_msg = f"{context} 연결 오류"
        elif isinstance(error, requests.exceptions.HTTPError):
            error_msg = f"{context} HTTP 오류: {error.response.status_code}"
        else:
            error_msg = f"{context} 알 수 없는 오류: {str(error)}"
        
        self.logger.error(error_msg)
        return error_msg
    
    def handle_data_error(self, error: Exception, context: str = "데이터 처리"):
        """데이터 처리 에러 처리"""
        if isinstance(error, (KeyError, IndexError)):
            error_msg = f"{context} 데이터 구조 오류: {str(error)}"
        elif isinstance(error, ValueError):
            error_msg = f"{context} 값 오류: {str(error)}"
        elif isinstance(error, TypeError):
            error_msg = f"{context} 타입 오류: {str(error)}"
        else:
            error_msg = f"{context} 처리 오류: {str(error)}"
        
        self.logger.error(error_msg)
        return error_msg

# 전역 인스턴스
cache = DataCache()
config = ConfigManager()
monitor = PerformanceMonitor()
error_handler = ErrorHandler()

# 편의 함수들
def get_config(key: str, default: Any = None) -> Any:
    """설정값 조회 편의 함수"""
    return config.get(key, default)

def set_config(key: str, value: Any) -> bool:
    """설정값 변경 편의 함수"""
    return config.set(key, value)

def log_performance(func_name: str = None):
    """성능 측정 데코레이터 편의 함수"""
    return monitor.timing_decorator(func_name)

def validate_data(data: Union[Dict, List[Dict]]) -> Dict:
    """데이터 검증 편의 함수"""
    if isinstance(data, dict):
        is_valid, errors = DataValidator.validate_article(data)
        return {
            'is_valid': is_valid,
            'errors': errors
        }
    else:
        return DataValidator.validate_articles_batch(data)

# 초기화 함수
def initialize_system():
    """시스템 초기화"""
    logger = setup_logging()
    logger.info("글로벌 기술 이슈 분석 시스템 초기화 시작")
    
    # 필요한 디렉토리 생성
    for directory in ['data', 'cache', 'config', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    # 기본 설정 파일 생성
    if not Path('config/settings.json').exists():
        config._save_config()
        logger.info("기본 설정 파일 생성 완료")
    
    logger.info("시스템 초기화 완료")

if __name__ == "__main__":
    # 시스템 초기화 테스트
    initialize_system()
    
    # 성능 모니터링 테스트
    @log_performance("test_function")
    def test_function():
        time.sleep(0.1)
        return "테스트 완료"
    
    result = test_function()
    print(monitor.get_report())
    
    # 설정 관리 테스트
    print("기본 국가 설정:", get_config("ui.default_countries"))
    
    # 데이터 검증 테스트
    test_article = {
        'title': '테스트 기사',
        'content': '테스트 내용',
        'country': '한국',
        'category': 'AI/ML',
        'date': '2025-07-05',
        'source': '테스트 소스',
        'sentiment': 0.5
    }
    
    validation_result = validate_data(test_article)
    print("데이터 검증 결과:", validation_result)
