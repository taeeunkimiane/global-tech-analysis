import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
import json
from collections import Counter
import networkx as nx
import plotly.figure_factory as ff

# 페이지 설정
st.set_page_config(
    page_title="글로벌 기술 이슈 분석 시스템",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .tech-category {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .country-flag {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 기술 카테고리 정의 (문서 기반)
TECH_CATEGORIES = {
    "하드웨어 혁신": {
        "keywords": ["스핀트로닉스", "마그노닉스", "양자컴퓨팅", "반도체", "칩셋", "프로세서"],
        "color": "#FF6B6B"
    },
    "AI/ML": {
        "keywords": ["인공지능", "머신러닝", "딥러닝", "GPT", "LLM", "베이지안"],
        "color": "#4ECDC4"
    },
    "보안/해킹": {
        "keywords": ["사이버보안", "해킹", "데이터보안", "프라이버시", "암호화"],
        "color": "#45B7D1"
    },
    "법률/규제": {
        "keywords": ["AI법", "데이터보호", "GDPR", "규제", "정책", "거버넌스"],
        "color": "#96CEB4"
    },
    "자율시스템": {
        "keywords": ["자율주행", "로봇", "드론", "IoT", "스마트시티"],
        "color": "#FFEAA7"
    }
}

COUNTRIES = {
    "미국": "🇺🇸",
    "중국": "🇨🇳", 
    "일본": "🇯🇵",
    "독일": "🇩🇪",
    "영국": "🇬🇧",
    "프랑스": "🇫🇷",
    "한국": "🇰🇷",
    "이스라엘": "🇮🇱",
    "싱가포르": "🇸🇬",
    "캐나다": "🇨🇦"
}

class TechNewsAnalyzer:
    def __init__(self):
        self.articles = []
        self.analysis_cache = {}
    
    def generate_sample_data(self):
        """샘플 데이터 생성 (실제 구현에서는 뉴스 스크래핑으로 대체)"""
        sample_articles = [
            {
                "title": "미국, 차세대 스핀트로닉스 칩 개발에 50억 달러 투자",
                "content": "미국 정부가 전자의 스핀을 이용한 차세대 컴퓨팅 기술인 스핀트로닉스 연구에 대규모 투자를 발표했다. 이는 중국과의 반도체 경쟁에서 우위를 점하기 위한 전략으로 분석된다.",
                "country": "미국",
                "date": "2025-07-01",
                "source": "Tech Times",
                "category": "하드웨어 혁신"
            },
            {
                "title": "EU, AI 법안 시행 1년 평가 보고서 발표",
                "content": "유럽연합이 AI 법안 시행 1년 후 평가 보고서를 발표했다. 고위험 AI 시스템에 대한 규제가 효과적으로 작동하고 있다고 평가했다.",
                "country": "독일",
                "date": "2025-06-28",
                "source": "Europe Tech",
                "category": "법률/규제"
            },
            {
                "title": "중국, 비모수 베이지안 AI 모델 개발 성공",
                "content": "중국 연구진이 불확실성을 정량적으로 표현할 수 있는 비모수 베이지안 AI 모델 개발에 성공했다고 발표했다. 이는 의료 진단 분야에 혁신을 가져올 것으로 기대된다.",
                "country": "중국", 
                "date": "2025-06-25",
                "source": "China AI News",
                "category": "AI/ML"
            },
            {
                "title": "일본 자율주행차, 마그노닉스 기술 적용",
                "content": "일본의 자동차 제조사가 마그노닉스 기술을 적용한 자율주행 시스템을 개발했다. 기존 대비 100배 빠른 연산 속도를 달성했다.",
                "country": "일본",
                "date": "2025-06-30",
                "source": "Nikkei Tech",
                "category": "자율시스템"
            },
            {
                "title": "한국, AI 보안 프레임워크 국제 표준 제안",
                "content": "한국이 AI 시스템의 보안 취약점을 체계적으로 평가하는 국제 표준 프레임워크를 제안했다. 적대적 공격과 데이터 오염 방어에 초점을 맞췄다.",
                "country": "한국",
                "date": "2025-07-03",
                "source": "Korea Herald",
                "category": "보안/해킹"
            }
        ]
        return sample_articles
    
    def analyze_sentiment(self, text):
        """감정 분석"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def categorize_article(self, article):
        """기사 카테고리 분류"""
        text = f"{article['title']} {article['content']}".lower()
        
        category_scores = {}
        for category, info in TECH_CATEGORIES.items():
            score = sum(1 for keyword in info['keywords'] if keyword.lower() in text)
            category_scores[category] = score
        
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return "기타"

def main():
    st.markdown('<h1 class="main-header">🌐 글로벌 기술 이슈 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI 시대의 기술 패러다임 변화와 국제적 거버넌스 동향 분석</p>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("## 🔧 분석 설정")
        
        # 날짜 범위 선택
        date_range = st.date_input(
            "분석 기간",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # 국가 선택
        selected_countries = st.multiselect(
            "분석 대상 국가",
            list(COUNTRIES.keys()),
            default=list(COUNTRIES.keys())[:5]
        )
        
        # 기술 카테고리 선택
        selected_categories = st.multiselect(
            "기술 카테고리",
            list(TECH_CATEGORIES.keys()),
            default=list(TECH_CATEGORIES.keys())
        )
        
        st.markdown("---")
        st.markdown("### 📊 실시간 업데이트")
        if st.button("뉴스 데이터 갱신"):
            st.success("데이터 갱신 완료!")
    
    # 메인 컨텐츠
    analyzer = TechNewsAnalyzer()
    articles = analyzer.generate_sample_data()
    
    # 필터링
    filtered_articles = [
        article for article in articles 
        if article['country'] in selected_countries
    ]
    
    # 상단 메트릭스
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>📰 총 기사 수</h3><h2>{len(filtered_articles)}</h2></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        unique_countries = len(set(article['country'] for article in filtered_articles))
        st.markdown(
            f'<div class="metric-card"><h3>🌍 분석 국가</h3><h2>{unique_countries}</h2></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        avg_sentiment = np.mean([analyzer.analyze_sentiment(article['content']) for article in filtered_articles])
        sentiment_emoji = "😊" if avg_sentiment > 0 else "😐" if avg_sentiment == 0 else "😟"
        st.markdown(
            f'<div class="metric-card"><h3>🎭 전체 감정</h3><h2>{sentiment_emoji} {avg_sentiment:.2f}</h2></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        categories = len(set(analyzer.categorize_article(article) for article in filtered_articles))
        st.markdown(
            f'<div class="metric-card"><h3>🔬 기술 분야</h3><h2>{categories}</h2></div>',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # 메인 차트들
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">📊 국가별 기술 이슈 분포</h3>', unsafe_allow_html=True)
        
        # 국가별 기사 수 계산
        country_counts = Counter([article['country'] for article in filtered_articles])
        
        fig = px.bar(
            x=list(country_counts.keys()),
            y=list(country_counts.values()),
            title="국가별 기사 수",
            color=list(country_counts.values()),
            color_continuous_scale="viridis"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">🔬 기술 카테고리별 분석</h3>', unsafe_allow_html=True)
        
        # 카테고리별 분포
        categories = [analyzer.categorize_article(article) for article in filtered_articles]
        category_counts = Counter(categories)
        
        colors = [TECH_CATEGORIES.get(cat, {}).get('color', '#888888') for cat in category_counts.keys()]
        
        fig = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="기술 분야별 비중",
            color_discrete_sequence=colors
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 시간별 트렌드 분석
    st.markdown('<h3 class="sub-header">📈 시간별 이슈 트렌드</h3>', unsafe_allow_html=True)
    
    # 날짜별 데이터 준비 (샘플)
    dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
    trend_data = []
    
    for date in dates:
        for category in TECH_CATEGORIES.keys():
            # 랜덤 샘플 데이터 (실제로는 실제 기사 데이터 사용)
            count = np.random.poisson(3) if category in selected_categories else 0
            trend_data.append({
                'date': date,
                'category': category,
                'count': count
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig = px.line(
        trend_df, 
        x='date', 
        y='count', 
        color='category',
        title="일별 기술 이슈 발생 트렌드",
        color_discrete_map={cat: info['color'] for cat, info in TECH_CATEGORIES.items()}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # 국가-기술 매트릭스
    st.markdown('<h3 class="sub-header">🌐 국가별 기술 분야 집중도</h3>', unsafe_allow_html=True)
    
    # 매트릭스 데이터 생성
    matrix_data = []
    for country in selected_countries:
        country_articles = [a for a in filtered_articles if a['country'] == country]
        for category in TECH_CATEGORIES.keys():
            category_count = sum(1 for a in country_articles if analyzer.categorize_article(a) == category)
            matrix_data.append({
                'country': f"{COUNTRIES.get(country, '')} {country}",
                'category': category,
                'count': category_count
            })
    
    matrix_df = pd.DataFrame(matrix_data)
    pivot_matrix = matrix_df.pivot(index='country', columns='category', values='count').fillna(0)
    
    fig = px.imshow(
        pivot_matrix.values,
        labels=dict(x="기술 분야", y="국가", color="기사 수"),
        x=pivot_matrix.columns,
        y=pivot_matrix.index,
        color_continuous_scale="Blues",
        title="국가-기술분야 히트맵"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # 최신 기사 목록
    st.markdown('<h3 class="sub-header">📰 최신 기술 이슈</h3>', unsafe_allow_html=True)
    
    for article in filtered_articles[:5]:
        with st.expander(f"{COUNTRIES.get(article['country'], '')} {article['title']}"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(article['content'])
            
            with col2:
                category = analyzer.categorize_article(article)
                color = TECH_CATEGORIES.get(category, {}).get('color', '#888888')
                st.markdown(
                    f'<span class="tech-category" style="background-color: {color}">{category}</span>',
                    unsafe_allow_html=True
                )
                st.write(f"**출처:** {article['source']}")
            
            with col3:
                sentiment = analyzer.analyze_sentiment(article['content'])
                sentiment_emoji = "😊" if sentiment > 0 else "😐" if sentiment == 0 else "😟"
                st.write(f"**감정:** {sentiment_emoji} {sentiment:.2f}")
                st.write(f"**날짜:** {article['date']}")
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>🔬 <strong>국제 시사 탐구 동아리</strong> - 글로벌 기술 이슈 분석 프로젝트</p>
            <p>데이터 기반 의사결정과 미래 기술 트렌드 예측을 통한 통찰력 개발</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
