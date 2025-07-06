# app.py - 메인 Streamlit 애플리케이션
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import os
from collections import Counter
import time

# 커스텀 모듈 import
try:
    from news_scraper import GlobalTechNewsScraper
    from advanced_analytics import TechTrendAnalyzer, AdvancedVisualizer
except ImportError:
    st.error("필수 모듈을 찾을 수 없습니다. news_scraper.py와 advanced_analytics.py 파일이 필요합니다.")
    st.stop()

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
        border-bottom: 2px solid #E8F4FD;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(45deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .tech-category {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .country-flag {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    .insight-box {
        background: linear-gradient(45deg, #e3f2fd, #f3e5f5);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .trend-up {
        color: #4caf50;
        font-weight: bold;
    }
    .trend-down {
        color: #f44336;
        font-weight: bold;
    }
    .data-source {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 캐싱을 위한 함수들
@st.cache_data(ttl=3600)  # 1시간 캐시
def load_cached_data():
    """캐시된 데이터 로드"""
    try:
        with open('tech_news_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data(ttl=1800)  # 30분 캐시
def get_sample_data():
    """샘플 데이터 생성 (실제 데이터가 없을 때)"""
    countries = ["미국", "중국", "일본", "독일", "영국", "프랑스", "한국", "이스라엘", "싱가포르", "캐나다"]
    categories = ["하드웨어 혁신", "AI/ML", "보안/해킹", "법률/규제", "자율시스템"]
    
    sample_articles = []
    base_date = datetime.now() - timedelta(days=30)
    
    titles_templates = {
        "하드웨어 혁신": [
            "{country}, 차세대 스핀트로닉스 칩 개발 성공",
            "{country} 연구진, 마그노닉스 기술로 연산속도 100배 향상",
            "{country} 반도체 기업, 양자컴퓨팅 칩 상용화 발표"
        ],
        "AI/ML": [
            "{country} AI 기업, 비모수 베이지안 모델 개발",
            "{country}, GPT 수준의 자체 언어모델 공개",
            "{country} 연구진, 설명가능한 AI 기술 돌파구 마련"
        ],
        "보안/해킹": [
            "{country} 정부기관, 대규모 사이버 공격 받아",
            "{country}, AI 시스템 보안 프레임워크 발표",
            "{country} 기업, 데이터 유출 사건으로 논란"
        ],
        "법률/규제": [
            "{country}, AI 규제 법안 의회 통과",
            "{country} 정부, 데이터 보호 강화 정책 발표",
            "{country}, 자율주행차 안전 기준 제정"
        ],
        "자율시스템": [
            "{country} 자동차 업체, 완전자율주행 기술 시연",
            "{country}, 드론 배송 서비스 본격 도입",
            "{country} 스마트시티 프로젝트 1단계 완료"
        ]
    }
    
    for i in range(150):  # 150개 샘플 기사
        country = np.random.choice(countries)
        category = np.random.choice(categories)
        
        title_template = np.random.choice(titles_templates[category])
        title = title_template.format(country=country)
        
        # 감정점수는 카테고리에 따라 다르게 설정
        if category == "보안/해킹":
            sentiment = np.random.normal(-0.2, 0.3)
        elif category == "법률/규제":
            sentiment = np.random.normal(-0.1, 0.4)
        else:
            sentiment = np.random.normal(0.2, 0.3)
        
        sentiment = np.clip(sentiment, -1, 1)
        
        article = {
            'title': title,
            'content': f"{title}과 관련된 상세 내용입니다. {category} 분야의 최신 동향을 다루고 있습니다.",
            'country': country,
            'category': category,
            'date': (base_date + timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
            'sentiment': round(sentiment, 3),
            'source': f"{country} Tech News",
            'url': f"https://example.com/article_{i}",
            'scraped_at': datetime.now().isoformat()
        }
        sample_articles.append(article)
    
    return sample_articles

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🌐 글로벌 기술 이슈 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">'
        'AI 시대의 기술 패러다임 변화와 국제적 거버넌스 동향 분석'
        '</p>', 
        unsafe_allow_html=True
    )
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("## 🔧 분석 설정")
        
        # 데이터 소스 선택
        data_source = st.radio(
            "데이터 소스",
            ["실시간 수집", "캐시된 데이터", "샘플 데이터"],
            index=2,  # 기본값: 샘플 데이터
            help="실시간 수집은 시간이 오래 걸릴 수 있습니다."
        )
        
        # 날짜 범위 선택
        date_range = st.date_input(
            "분석 기간",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now(),
            help="분석할 기간을 선택해주세요."
        )
        
        # 국가 선택
        countries = ["미국", "중국", "일본", "독일", "영국", "프랑스", "한국", "이스라엘", "싱가포르", "캐나다"]
        selected_countries = st.multiselect(
            "분석 대상 국가",
            countries,
            default=countries[:5],
            help="분석할 국가들을 선택해주세요."
        )
        
        # 기술 카테고리 선택
        categories = ["하드웨어 혁신", "AI/ML", "보안/해킹", "법률/규제", "자율시스템"]
        selected_categories = st.multiselect(
            "기술 카테고리",
            categories,
            default=categories,
            help="분석할 기술 분야를 선택해주세요."
        )
        
        st.markdown("---")
        
        # 데이터 갱신
        st.markdown("### 📊 데이터 관리")
        
        if st.button("🔄 데이터 갱신", type="primary"):
            if data_source == "실시간 수집":
                with st.spinner("뉴스 데이터 수집 중... (약 2-3분 소요)"):
                    scraper = GlobalTechNewsScraper()
                    articles = scraper.scrape_all_sources()
                    scraper.save_to_json(articles)
                    st.success(f"✅ {len(articles)}개 기사 수집 완료!")
                    st.experimental_rerun()
            else:
                st.success("✅ 데이터 갱신 완료!")
        
        # 데이터 내보내기
        if st.button("📥 데이터 다운로드"):
            # 현재 분석 데이터를 CSV로 내보내기
            if 'current_articles' in locals():
                df = pd.DataFrame(current_articles)
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="CSV 다운로드",
                    data=csv,
                    file_name=f"tech_news_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        if data_source == "실시간 수집":
            scraper = GlobalTechNewsScraper()
            current_articles = scraper.scrape_all_sources()
        elif data_source == "캐시된 데이터":
            cached_data = load_cached_data()
            if cached_data:
                current_articles = cached_data
            else:
                st.warning("캐시된 데이터가 없습니다. 샘플 데이터를 사용합니다.")
                current_articles = get_sample_data()
        else:  # 샘플 데이터
            current_articles = get_sample_data()
    
    # 데이터 필터링
    filtered_articles = [
        article for article in current_articles 
        if (article['country'] in selected_countries and 
            article['category'] in selected_categories)
    ]
    
    if not filtered_articles:
        st.error("선택한 조건에 맞는 데이터가 없습니다. 필터 조건을 확인해주세요.")
        return
    
    # 분석기 초기화
    analyzer = TechTrendAnalyzer(filtered_articles)
    visualizer = AdvancedVisualizer(analyzer)
    
    # 상단 메트릭스
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            f'''<div class="metric-card">
                <h4>📰 총 기사 수</h4>
                <h2 style="color: #667eea;">{len(filtered_articles):,}</h2>
                <small>선택된 조건의 기사</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col2:
        unique_countries = len(set(article['country'] for article in filtered_articles))
        st.markdown(
            f'''<div class="metric-card">
                <h4>🌍 분석 국가</h4>
                <h2 style="color: #4ECDC4;">{unique_countries}</h2>
                <small>개 국가</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col3:
        unique_categories = len(set(article['category'] for article in filtered_articles))
        st.markdown(
            f'''<div class="metric-card">
                <h4>🔬 기술 분야</h4>
                <h2 style="color: #45B7D1;">{unique_categories}</h2>
                <small>개 분야</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col4:
        avg_sentiment = np.mean([article['sentiment'] for article in filtered_articles])
        sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment >= -0.1 else "😟"
        sentiment_color = "#4CAF50" if avg_sentiment > 0.1 else "#FFC107" if avg_sentiment >= -0.1 else "#F44336"
        st.markdown(
            f'''<div class="metric-card">
                <h4>🎭 전체 감정</h4>
                <h2 style="color: {sentiment_color};">{sentiment_emoji} {avg_sentiment:.3f}</h2>
                <small>감정 점수</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col5:
        # 최신 기사 날짜
        latest_date = max(article['date'] for article in filtered_articles)
        st.markdown(
            f'''<div class="metric-card">
                <h4>📅 최신 업데이트</h4>
                <h2 style="color: #96CEB4; font-size: 1.2rem;">{latest_date}</h2>
                <small>마지막 기사</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # 탭 메뉴
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 개요 대시보드", 
        "🌍 국가별 분석", 
        "🔬 기술 트렌드", 
        "🕸️ 연관성 분석", 
        "📈 심화 분석"
    ])
    
    with tab1:
        st.markdown('<h3 class="sub-header">📊 전체 현황 대시보드</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 국가별 기사 수 분포
            country_counts = Counter([article['country'] for article in filtered_articles])
            
            fig_country = px.bar(
                x=list(country_counts.keys()),
                y=list(country_counts.values()),
                title="국가별 기술 이슈 기사 수",
                color=list(country_counts.values()),
                color_continuous_scale="viridis",
                text=list(country_counts.values())
            )
            fig_country.update_traces(texttemplate='%{text}', textposition='outside')
            fig_country.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            # 기술 카테고리별 분포
            category_counts = Counter([article['category'] for article in filtered_articles])
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            fig_category = px.pie(
                values=list(category_counts.values()),
                names=list(category_counts.keys()),
                title="기술 분야별 관심도",
                color_discrete_sequence=colors
            )
            fig_category.update_traces(textposition='inside', textinfo='percent+label')
            fig_category.update_layout(height=400)
            st.plotly_chart(fig_category, use_container_width=True)
        
        # 시간별 트렌드
        st.markdown('<h4 class="sub-header">📈 시간별 이슈 발생 트렌드</h4>', unsafe_allow_html=True)
        
        # 날짜별 기사 수 계산
        df_articles = pd.DataFrame(filtered_articles)
        df_articles['date'] = pd.to_datetime(df_articles['date'])
        
        daily_counts = df_articles.groupby(['date', 'category']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            daily_counts, 
            x='date', 
            y='count', 
            color='category',
            title="일별 기술 이슈 발생 트렌드",
            color_discrete_sequence=colors
        )
        fig_timeline.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # 인사이트 박스
        insights = analyzer.generate_insights_report()
        st.markdown(
            f'''<div class="insight-box">
                <h4>🧠 주요 인사이트</h4>
                <pre style="white-space: pre-wrap; font-family: inherit; font-size: 0.9rem;">{insights}</pre>
            </div>''',
            unsafe_allow_html=True
        )
    
    with tab2:
        st.markdown('<h3 class="sub-header">🌍 국가별 기술 이슈 분석</h3>', unsafe_allow_html=True)
        
        # 국가-기술 매트릭스
        matrix_data = []
        for country in selected_countries:
            country_articles = [a for a in filtered_articles if a['country'] == country]
            for category in selected_categories:
                category_count = sum(1 for a in country_articles if a['category'] == category)
                matrix_data.append({
                    'country': country,
                    'category': category,
                    'count': category_count
                })
        
        matrix_df = pd.DataFrame(matrix_data)
        
        if not matrix_df.empty:
            pivot_matrix = matrix_df.pivot(index='country', columns='category', values='count').fillna(0)
            
            fig_heatmap = px.imshow(
                pivot_matrix.values,
                labels=dict(x="기술 분야", y="국가", color="기사 수"),
                x=pivot_matrix.columns,
                y=pivot_matrix.index,
                color_continuous_scale="Blues",
                title="국가별 기술 분야 집중도 히트맵",
                text_auto=True
            )
            fig_heatmap.update_layout(height=600)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 국가별 감정 분석
        col1, col2 = st.columns(2)
        
        with col1:
            country_sentiment = df_articles.groupby('country')['sentiment'].mean().sort_values(ascending=False)
            
            fig_sentiment = px.bar(
                x=country_sentiment.index,
                y=country_sentiment.values,
                title="국가별 평균 감정 점수",
                color=country_sentiment.values,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0
            )
            fig_sentiment.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="중립선")
            fig_sentiment.update_layout(height=400)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with col2:
            # 국가별 기사 수와 감정의 관계
            country_stats = df_articles.groupby('country').agg({
                'sentiment': 'mean',
                'title': 'count'
            }).rename(columns={'title': 'article_count'}).reset_index()
            
            fig_scatter = px.scatter(
                country_stats,
                x='article_count',
                y='sentiment',
                size='article_count',
                color='sentiment',
                hover_name='country',
                title="기사 수 vs 감정 점수",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="black")
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.markdown('<h3 class="sub-header">🔬 기술 트렌드 분석</h3>', unsafe_allow_html=True)
        
        # 신흥 트렌드 키워드
        col1, col2 = st.columns([2, 1])
        
        with col1:
            trending_keywords = analyzer.detect_emerging_trends()
            
            if trending_keywords:
                trend_df = pd.DataFrame([
                    {
                        'keyword': keyword,
                        'trend_score': data['trend_score'],
                        'recent_freq': data['recent_frequency'],
                        'older_freq': data['older_frequency']
                    }
                    for keyword, data in list(trending_keywords.items())[:15]
                ])
                
                fig_trend = px.bar(
                    trend_df,
                    x='trend_score',
                    y='keyword',
                    orientation='h',
                    title="신흥 트렌드 키워드 (최근 7일 vs 이전 기간)",
                    color='trend_score',
                    color_continuous_scale="Reds",
                    hover_data=['recent_freq', 'older_freq']
                )
                fig_trend.update_layout(height=500)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("충분한 시계열 데이터가 없어 트렌드 분석을 수행할 수 없습니다.")
        
        with col2:
            st.markdown("### 📊 트렌드 해석")
            st.markdown("""
            **트렌드 점수 해석:**
            - 2.0x 이상: 급상승 트렌드 🔥
            - 1.5x - 2.0x: 상승 트렌드 📈
            - 1.0x - 1.5x: 완만한 증가 ↗️
            
            **주의사항:**
            - 최소 3회 이상 언급된 키워드만 포함
            - 최근 7일 vs 이전 기간 비교
            """)
            
            if trending_keywords:
                st.markdown("### 🔥 핫 키워드")
                for keyword, data in list(trending_keywords.items())[:5]:
                    st.markdown(f"**{keyword}**")
                    st.markdown(f"📈 {data['trend_score']:.1f}x 증가")
                    st.markdown("---")
        
        # 감정 트렌드 분석
        st.markdown('<h4 class="sub-header">🎭 기술 분야별 감정 트렌드</h4>', unsafe_allow_html=True)
        
        sentiment_fig = visualizer.create_sentiment_timeline()
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h3 class="sub-header">🕸️ 기술 간 연관성 네트워크</h3>', unsafe_allow_html=True)
        
        # 네트워크 시각화
        network_fig = visualizer.create_network_visualization()
        st.plotly_chart(network_fig, use_container_width=True)
        
        # 토픽 모델링 결과
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 주요 토픽 분석")
            
            try:
                topics_result = analyzer.get_trending_topics()
                
                for topic_name, topic_info in topics_result['topics'].items():
                    with st.expander(f"📌 {topic_name}"):
                        words = topic_info['words'][:8]
                        weights = topic_info['weight'][:8]
                        
                        topic_df = pd.DataFrame({
                            'word': words,
                            'weight': weights
                        })
                        
                        fig_topic = px.bar(
                            topic_df,
                            x='weight',
                            y='word',
                            orientation='h',
                            title=f"{topic_name} 주요 키워드"
                        )
                        fig_topic.update_layout(height=300)
                        st.plotly_chart(fig_topic, use_container_width=True)
            
            except Exception as e:
                st.warning(f"토픽 분석 중 오류 발생: {str(e)}")
        
        with col2:
            st.markdown("### 🔗 기술 연관성 인사이트")
            
            # 국가별 기술 특화도
            country_focus = analyzer.analyze_country_tech_focus()
            
            for country, tech_dist in list(country_focus.items())[:5]:
                if tech_dist:
                    top_tech = max(tech_dist, key=tech_dist.get)
                    focus_rate = tech_dist[top_tech]
                    
                    st.markdown(
                        f"""
                        **{country}**
                        - 특화 분야: {top_tech}
                        - 집중도: {focus_rate:.1%}
                        """
                    )
                    
                    # 미니 차트
                    tech_names = list(tech_dist.keys())
                    tech_values = list(tech_dist.values())
                    
                    if len(tech_names) > 1:
                        mini_fig = px.pie(
                            values=tech_values,
                            names=tech_names,
                            title=f"{country} 기술 분포",
                            height=200
                        )
                        mini_fig.update_traces(textposition='inside', textinfo='percent')
                        mini_fig.update_layout(showlegend=False)
                        st.plotly_chart(mini_fig, use_container_width=True)
    
    with tab5:
        st.markdown('<h3 class="sub-header">📈 심화 분석 및 예측</h3>', unsafe_allow_html=True)
        
        # 추가 분석 옵션
        analysis_type = st.selectbox(
            "분석 유형 선택",
            ["감정 변화 패턴", "국가 간 기술 경쟁도", "이슈 생명주기 분석", "키워드 동시출현 분석"]
        )
        
        if analysis_type == "감정 변화 패턴":
            st.markdown("### 📊 기술 분야별 감정 변화 패턴")
            
            # 기간별 감정 변화
            df_articles['week'] = df_articles['date'].dt.isocalendar().week
            weekly_sentiment = df_articles.groupby(['week', 'category'])['sentiment'].mean().reset_index()
            
            fig_sentiment_trend = px.line(
                weekly_sentiment,
                x='week',
                y='sentiment',
                color='category',
                title="주별 기술 분야 감정 변화",
                markers=True
            )
            fig_sentiment_trend.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_sentiment_trend, use_container_width=True)
        
        elif analysis_type == "국가 간 기술 경쟁도":
            st.markdown("### 🏆 국가별 기술 경쟁력 매트릭스")
            
            # 국가별 기술 분야 점유율
            competition_matrix = []
            for category in selected_categories:
                category_articles = [a for a in filtered_articles if a['category'] == category]
                total_articles = len(category_articles)
                
                for country in selected_countries:
                    country_articles = [a for a in category_articles if a['country'] == country]
                    share = len(country_articles) / total_articles if total_articles > 0 else 0
                    
                    competition_matrix.append({
                        'country': country,
                        'category': category,
                        'share': share,
                        'article_count': len(country_articles)
                    })
            
            comp_df = pd.DataFrame(competition_matrix)
            
            if not comp_df.empty:
                pivot_comp = comp_df.pivot(index='country', columns='category', values='share').fillna(0)
                
                fig_competition = px.imshow(
                    pivot_comp.values * 100,  # 퍼센트로 변환
                    labels=dict(x="기술 분야", y="국가", color="점유율(%)"),
                    x=pivot_comp.columns,
                    y=pivot_comp.index,
                    color_continuous_scale="RdYlBu_r",
                    title="국가별 기술 분야 점유율 매트릭스",
                    text_auto=".1f"
                )
                fig_competition.update_layout(height=500)
                st.plotly_chart(fig_competition, use_container_width=True)
        
        elif analysis_type == "이슈 생명주기 분석":
            st.markdown("### ⏳ 기술 이슈 생명주기 분석")
            
            # 이슈의 지속 기간과 강도 분석
            lifecycle_data = []
            for category in selected_categories:
                cat_articles = [a for a in filtered_articles if a['category'] == category]
                if cat_articles:
                    dates = [datetime.strptime(a['date'], '%Y-%m-%d') for a in cat_articles]
                    duration = (max(dates) - min(dates)).days
                    intensity = len(cat_articles) / max(duration, 1)  # 일당 기사 수
                    avg_sentiment = np.mean([a['sentiment'] for a in cat_articles])
                    
                    lifecycle_data.append({
                        'category': category,
                        'duration_days': duration,
                        'intensity': intensity,
                        'avg_sentiment': avg_sentiment,
                        'total_articles': len(cat_articles)
                    })
            
            if lifecycle_data:
                lifecycle_df = pd.DataFrame(lifecycle_data)
                
                fig_lifecycle = px.scatter(
                    lifecycle_df,
                    x='duration_days',
                    y='intensity',
                    size='total_articles',
                    color='avg_sentiment',
                    hover_name='category',
                    title="기술 이슈 생명주기: 지속기간 vs 강도",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0
                )
                fig_lifecycle.update_layout(height=500)
                st.plotly_chart(fig_lifecycle, use_container_width=True)
        
        else:  # 키워드 동시출현 분석
            st.markdown("### 🔗 키워드 동시출현 네트워크")
            
            # 키워드 추출 및 동시출현 분석
            from collections import defaultdict
            import itertools
            
            cooccurrence = defaultdict(int)
            
            # 각 기사에서 키워드 추출
            tech_keywords = [
                'AI', '인공지능', '머신러닝', '딥러닝',
                '양자컴퓨팅', '블록체인', '자율주행', 'IoT',
                '5G', '로봇', '드론', '스마트시티',
                '사이버보안', '해킹', '프라이버시'
            ]
            
            for article in filtered_articles:
                text = article['title'] + ' ' + article['content']
                found_keywords = [kw for kw in tech_keywords if kw.lower() in text.lower()]
                
                # 키워드 쌍의 동시출현 카운트
                for pair in itertools.combinations(found_keywords, 2):
                    sorted_pair = tuple(sorted(pair))
                    cooccurrence[sorted_pair] += 1
            
            # 상위 동시출현 쌍 시각화
            if cooccurrence:
                cooc_data = [
                    {'keyword1': pair[0], 'keyword2': pair[1], 'count': count}
                    for pair, count in sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:20]
                ]
                
                cooc_df = pd.DataFrame(cooc_data)
                cooc_df['pair'] = cooc_df['keyword1'] + ' ↔ ' + cooc_df['keyword2']
                
                fig_cooc = px.bar(
                    cooc_df,
                    x='count',
                    y='pair',
                    orientation='h',
                    title="키워드 동시출현 빈도 (Top 20)",
                    color='count',
                    color_continuous_scale="Viridis"
                )
                fig_cooc.update_layout(height=600)
                st.plotly_chart(fig_cooc, use_container_width=True)
            else:
                st.info("충분한 키워드 동시출현 데이터가 없습니다.")
    
    # 최신 기사 피드
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📰 최신 기술 이슈 피드</h3>', unsafe_allow_html=True)
    
    # 최신 기사 5개 표시
    latest_articles = sorted(filtered_articles, key=lambda x: x['date'], reverse=True)[:10]
    
    for i, article in enumerate(latest_articles):
        with st.expander(f"📄 {article['title']}", expanded=(i < 3)):  # 처음 3개는 펼쳐진 상태
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(article['content'])
                
                # 기사 URL 링크
                if article.get('url'):
                    st.markdown(f"🔗 [원문 보기]({article['url']})")
            
            with col2:
                # 카테고리 배지
                category_colors = {
                    "하드웨어 혁신": "#FF6B6B",
                    "AI/ML": "#4ECDC4", 
                    "보안/해킹": "#45B7D1",
                    "법률/규제": "#96CEB4",
                    "자율시스템": "#FFEAA7"
                }
                color = category_colors.get(article['category'], '#888888')
                st.markdown(
                    f'<span class="tech-category" style="background-color: {color}">{article["category"]}</span>',
                    unsafe_allow_html=True
                )
                
                st.write(f"**🌍 국가:** {article['country']}")
                st.write(f"**📅 날짜:** {article['date']}")
                st.write(f"**📰 출처:** {article['source']}")
            
            with col3:
                # 감정 분석 결과
                sentiment = article['sentiment']
                if sentiment > 0.1:
                    sentiment_emoji = "😊"
                    sentiment_color = "#4CAF50"
                    sentiment_text = "긍정적"
                elif sentiment < -0.1:
                    sentiment_emoji = "😟"
                    sentiment_color = "#F44336"
                    sentiment_text = "부정적"
                else:
                    sentiment_emoji = "😐"
                    sentiment_color = "#FFC107"
                    sentiment_text = "중립적"
                
                st.markdown(f"**🎭 감정 분석**")
                st.markdown(
                    f'<div style="text-align: center; color: {sentiment_color};">'
                    f'<div style="font-size: 2rem;">{sentiment_emoji}</div>'
                    f'<div>{sentiment_text}</div>'
                    f'<div style="font-size: 0.9rem;">({sentiment:.3f})</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # 푸터
    st.markdown("---")
    st.markdown(
        f'''<div class="data-source">
            <p>📊 <strong>데이터 소스:</strong> {data_source} | 
            🔄 <strong>마지막 업데이트:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            📈 <strong>분석 기사 수:</strong> {len(filtered_articles):,}개</p>
        </div>''',
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
            <h4>🔬 국제 시사 탐구 동아리 - 글로벌 기술 이슈 분석 프로젝트</h4>
            <p>데이터 기반 의사결정과 미래 기술 트렌드 예측을 통한 통찰력 개발</p>
            <p><small>Made with ❤️ using Streamlit & Python</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
