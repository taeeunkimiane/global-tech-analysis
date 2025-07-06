# app.py - 글로벌 기술 이슈 분석 시스템 (독립형 버전)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
import time
import requests

# 선택적 import (없어도 작동하도록)
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    .data-source {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 기술 카테고리 및 국가 정의
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
    """간소화된 기술 뉴스 분석기"""
    
    def __init__(self, articles):
        self.articles = articles
        self.df = pd.DataFrame(articles) if articles else pd.DataFrame()
        if not self.df.empty:
            self.df['date'] = pd.to_datetime(self.df['date'])
    
    def analyze_sentiment(self, text):
        """감정 분석 (TextBlob 사용 또는 간단한 키워드 기반)"""
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                return blob.sentiment.polarity
            except:
                pass
        
        # 간단한 키워드 기반 감정 분석
        positive_words = ['성공', '혁신', '개발', '발전', '향상', 'breakthrough', 'success', 'innovation']
        negative_words = ['공격', '위험', '문제', '실패', '우려', 'attack', 'threat', 'problem', 'failure']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 0.3
        elif neg_count > pos_count:
            return -0.3
        else:
            return 0.0
    
    def categorize_article(self, article):
        """기사 카테고리 분류"""
        text = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        category_scores = {}
        for category, info in TECH_CATEGORIES.items():
            score = sum(1 for keyword in info['keywords'] if keyword.lower() in text)
            category_scores[category] = score
        
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        return "기타"
    
    def detect_emerging_trends(self, window_days=7):
        """신흥 트렌드 감지"""
        if self.df.empty:
            return {}
        
        recent_date = self.df['date'].max()
        cutoff_date = recent_date - timedelta(days=window_days)
        
        recent_articles = self.df[self.df['date'] >= cutoff_date]
        older_articles = self.df[self.df['date'] < cutoff_date]
        
        # 간단한 키워드 빈도 분석
        recent_text = ' '.join(recent_articles['title'].fillna('') + ' ' + recent_articles['content'].fillna(''))
        older_text = ' '.join(older_articles['title'].fillna('') + ' ' + older_articles['content'].fillna(''))
        
        # 키워드 추출 (간단 버전)
        keywords = ['AI', 'blockchain', 'quantum', 'autonomous', 'cybersecurity', 'regulation']
        trending = {}
        
        for keyword in keywords:
            recent_count = recent_text.lower().count(keyword.lower())
            older_count = older_text.lower().count(keyword.lower()) or 1
            
            if recent_count >= 2:
                trend_score = recent_count / older_count
                if trend_score > 1.5:
                    trending[keyword] = {
                        'recent_frequency': recent_count,
                        'older_frequency': older_count,
                        'trend_score': trend_score
                    }
        
        return dict(sorted(trending.items(), key=lambda x: x[1]['trend_score'], reverse=True))
    
    def analyze_country_tech_focus(self):
        """국가별 기술 집중도 분석"""
        if self.df.empty:
            return {}
        
        country_tech = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            country_tech[row['country']][row['category']] += 1
        
        # 정규화
        country_tech_normalized = {}
        for country, tech_counts in country_tech.items():
            total_articles = sum(tech_counts.values())
            if total_articles > 0:
                country_tech_normalized[country] = {
                    tech: count / total_articles
                    for tech, count in tech_counts.items()
                }
        
        return country_tech_normalized
    
    def generate_insights_report(self):
        """인사이트 리포트 생성"""
        if self.df.empty:
            return "데이터가 없습니다."
        
        total_articles = len(self.df)
        date_range = f"{self.df['date'].min().strftime('%Y-%m-%d')} ~ {self.df['date'].max().strftime('%Y-%m-%d')}"
        
        report = []
        report.append(f"📊 **분석 기간**: {date_range}")
        report.append(f"📰 **총 기사 수**: {total_articles:,}개")
        report.append("")
        
        # 국가별 활동도
        country_counts = self.df['country'].value_counts()
        report.append("🌍 **국가별 기술 이슈 활동도**")
        for country, count in country_counts.head(5).items():
            percentage = (count / total_articles) * 100
            flag = COUNTRIES.get(country, "🏳️")
            report.append(f"- {flag} {country}: {count}개 ({percentage:.1f}%)")
        report.append("")
        
        # 기술 분야별 관심도
        category_counts = self.df['category'].value_counts()
        report.append("🔬 **기술 분야별 관심도**")
        for category, count in category_counts.head(5).items():
            percentage = (count / total_articles) * 100
            report.append(f"- {category}: {count}개 ({percentage:.1f}%)")
        report.append("")
        
        # 평균 감정
        avg_sentiment = self.df['sentiment'].mean()
        sentiment_label = "긍정적" if avg_sentiment > 0.1 else "부정적" if avg_sentiment < -0.1 else "중립적"
        report.append("🎭 **전반적 여론**")
        report.append(f"- 평균 감정 점수: {avg_sentiment:.3f} ({sentiment_label})")
        
        return "\n".join(report)

@st.cache_data
def load_sample_data():
    """향상된 샘플 데이터 생성"""
    countries = list(COUNTRIES.keys())
    categories = list(TECH_CATEGORIES.keys())
    
    # 실제적인 샘플 기사 데이터
    sample_articles = [
        {
            'title': '미국, 차세대 스핀트로닉스 칩 개발 성공',
            'content': '미국 연구진이 전자의 스핀을 이용한 혁신적인 컴퓨팅 기술을 개발했다. 이 기술은 기존 반도체보다 100배 빠른 연산 속도와 1/10 수준의 전력 소비를 실현한다.',
            'country': '미국',
            'category': '하드웨어 혁신',
            'date': '2025-07-03',
            'sentiment': 0.752,
            'source': 'MIT Technology Review',
            'url': 'https://example.com/article/1'
        },
        {
            'title': '중국 AI 기업, 비모수 베이지안 모델 개발',
            'content': '중국 AI 기업이 불확실성을 정량적으로 표현할 수 있는 비모수 베이지안 AI 모델을 개발했다고 발표했다.',
            'country': '중국',
            'category': 'AI/ML',
            'date': '2025-07-02',
            'sentiment': 0.634,
            'source': 'Tech in Asia',
            'url': 'https://example.com/article/2'
        },
        {
            'title': '일본 정부기관, 대규모 사이버 공격 받아',
            'content': '일본의 주요 정부기관이 AI 시스템을 대상으로 한 적대적 공격을 받았다고 발표했다.',
            'country': '일본',
            'category': '보안/해킹',
            'date': '2025-07-01',
            'sentiment': -0.423,
            'source': 'Nikkei Asia',
            'url': 'https://example.com/article/3'
        },
        {
            'title': '독일, AI 규제 법안 의회 통과',
            'content': '독일 의회가 AI 시스템의 안전성과 투명성을 보장하기 위한 포괄적인 법적 프레임워크를 통과시켰다.',
            'country': '독일',
            'category': '법률/규제',
            'date': '2025-06-30',
            'sentiment': 0.156,
            'source': 'The Next Web',
            'url': 'https://example.com/article/4'
        },
        {
            'title': '한국 자동차 업체, 완전자율주행 기술 시연',
            'content': '한국의 주요 자동차 제조사가 마그노닉스 기술을 적용한 완전자율주행 시스템을 성공적으로 시연했다.',
            'country': '한국',
            'category': '자율시스템',
            'date': '2025-06-29',
            'sentiment': 0.687,
            'source': 'TechCrunch',
            'url': 'https://example.com/article/5'
        }
    ]
    
    # 추가 샘플 데이터 생성
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        country = countries[i % len(countries)]
        category = categories[i % len(categories)]
        
        # 카테고리별 감정 점수 조정
        if category == "보안/해킹":
            sentiment = np.random.normal(-0.2, 0.3)
        elif category == "법률/규제":
            sentiment = np.random.normal(-0.1, 0.4)
        else:
            sentiment = np.random.normal(0.2, 0.3)
        
        sentiment = np.clip(sentiment, -1, 1)
        
        article = {
            'title': f'{country} {category} 관련 최신 기술 뉴스 {i+6}',
            'content': f'이것은 {category} 분야의 {country} 관련 기사입니다. 최신 기술 동향과 발전사항을 다루고 있습니다.',
            'country': country,
            'category': category,
            'date': (base_date + timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
            'sentiment': round(sentiment, 3),
            'source': f'{country} Tech News',
            'url': f'https://example.com/article/{i+6}'
        }
        sample_articles.append(article)
    
    return sample_articles

def create_network_visualization(analyzer):
    """네트워크 시각화 생성"""
    if not NETWORKX_AVAILABLE:
        fig = go.Figure()
        fig.add_annotation(
            text="NetworkX가 설치되어 있지 않습니다.<br>네트워크 분석 기능을 사용하려면 pip install networkx를 실행하세요.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        return fig
    
    # 간단한 기술 연관성 네트워크
    G = nx.Graph()
    categories = list(TECH_CATEGORIES.keys())
    
    # 노드 추가
    for cat in categories:
        G.add_node(cat)
    
    # 간단한 연결 추가 (예시)
    connections = [
        ("AI/ML", "자율시스템"),
        ("하드웨어 혁신", "AI/ML"),
        ("보안/해킹", "AI/ML"),
        ("법률/규제", "AI/ML"),
        ("법률/규제", "보안/해킹")
    ]
    
    for conn in connections:
        G.add_edge(conn[0], conn[1])
    
    # 레이아웃 계산
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # 엣지 그리기
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                           line=dict(width=2, color='#888'),
                           hoverinfo='none',
                           mode='lines')
    
    # 노드 그리기
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_colors.append(TECH_CATEGORIES[node]['color'])
    
    node_trace = go.Scatter(x=node_x, y=node_y,
                           mode='markers+text',
                           hoverinfo='text',
                           text=node_text,
                           textposition="middle center",
                           marker=dict(size=30, color=node_colors, line=dict(width=2, color='white')))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(title='기술 분야 간 연관성 네트워크',
                                   titlefont_size=16,
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=20,l=5,r=5,t=40),
                                   annotations=[ dict(
                                       text="기술 분야 간 연관성을 보여주는 네트워크",
                                       showarrow=False,
                                       xref="paper", yref="paper",
                                       x=0.005, y=-0.002 ) ],
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🌐 글로벌 기술 이슈 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">AI 시대의 기술 패러다임 변화와 국제적 거버넌스 동향 분석</p>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("## 🔧 분석 설정")
        
        # 성공 메시지
        st.success("🎉 배포 성공!")
        
        # 기능 상태 표시
        st.markdown("### 📊 기능 상태")
        st.write(f"✅ 기본 분석: 사용 가능")
        st.write(f"{'✅' if TEXTBLOB_AVAILABLE else '⚠️'} 감정 분석: {'고급' if TEXTBLOB_AVAILABLE else '기본'}")
        st.write(f"{'✅' if NETWORKX_AVAILABLE else '⚠️'} 네트워크 분석: {'사용 가능' if NETWORKX_AVAILABLE else '제한됨'}")
        st.write(f"{'✅' if SKLEARN_AVAILABLE else '⚠️'} 고급 분석: {'사용 가능' if SKLEARN_AVAILABLE else '제한됨'}")
        
        st.markdown("---")
        
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
        if st.button("🔄 데이터 갱신"):
            st.rerun()
    
    # 메인 컨텐츠
    with st.spinner("📊 데이터 로딩 중..."):
        articles = load_sample_data()
    
    # 데이터 필터링
    filtered_articles = [
        article for article in articles 
        if article['country'] in selected_countries and article['category'] in selected_categories
    ]
    
    if not filtered_articles:
        st.error("선택한 조건에 맞는 데이터가 없습니다. 필터 조건을 확인해주세요.")
        return
    
    # 분석기 초기화
    analyzer = TechNewsAnalyzer(filtered_articles)
    
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
        "📰 최신 기사"
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
            colors = [TECH_CATEGORIES[cat]['color'] for cat in category_counts.keys()]
            
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
        
        df_articles = pd.DataFrame(filtered_articles)
        df_articles['date'] = pd.to_datetime(df_articles['date'])
        daily_counts = df_articles.groupby(['date', 'category']).size().reset_index(name='count')
        
        colors = [TECH_CATEGORIES[cat]['color'] for cat in TECH_CATEGORIES.keys()]
        
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
                    'country': f"{COUNTRIES.get(country, '')} {country}",
                    'category': category,
                    'count': category_count
                })
        
        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
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
            # 국가별 기술 특화도
            country_focus = analyzer.analyze_country_tech_focus()
            
            st.markdown("### 🎯 국가별 기술 특화 분야")
            for country, tech_dist in list(country_focus.items())[:5]:
                if tech_dist:
                    top_tech = max(tech_dist, key=tech_dist.get)
                    focus_rate = tech_dist[top_tech]
                    flag = COUNTRIES.get(country, "🏳️")
                    
                    st.markdown(f"**{flag} {country}**")
                    st.markdown(f"특화 분야: {top_tech} ({focus_rate:.1%} 집중)")
                    
                    # 미니 차트
                    if len(tech_dist) > 1:
                        tech_names = list(tech_dist.keys())
                        tech_values = list(tech_dist.values())
                        
                        mini_fig = px.pie(
                            values=tech_values,
                            names=tech_names,
                            title=f"{country} 기술 분포",
                            height=200
                        )
                        mini_fig.update_traces(textposition='inside', textinfo='percent')
                        mini_fig.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(mini_fig, use_container_width=True)
                    
                    st.markdown("---")
    
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
                    for keyword, data in trending_keywords.items()
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
                fig_trend.update_layout(height=400)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("현재 데이터에서 뚜렷한 신흥 트렌드를 감지하지 못했습니다.")
        
        with col2:
            st.markdown("### 📊 트렌드 해석")
            st.markdown("""
            **트렌드 점수 해석:**
            - 2.0x 이상: 급상승 트렌드 🔥
            - 1.5x - 2.0x: 상승 트렌드 📈
            - 1.0x - 1.5x: 완만한 증가 ↗️
            
            **주의사항:**
            - 최소 2회 이상 언급된 키워드만 포함
            - 최근 7일 vs 이전 기간 비교
            """)
            
            if trending_keywords:
                st.markdown("### 🔥 핫 키워드")
                for keyword, data in list(trending_keywords.items())[:3]:
                    st.markdown(f"**{keyword}**")
                    st.markdown(f"📈 {data['trend_score']:.1f}x 증가")
                    st.markdown("---")
        
        # 감정 트렌드 분석
        st.markdown('<h4 class="sub-header">🎭 기술 분야별 감정 트렌드</h4>', unsafe_allow_html=True)
        
        category_sentiment = df_articles.groupby('category')['sentiment'].agg(['mean', 'count']).reset_index()
        
        fig_cat_sentiment = px.bar(
            category_sentiment,
            x='category',
            y='mean',
            title="기술 분야별 평균 감정 점수",
            color='mean',
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            hover_data=['count']
        )
        fig_cat_sentiment.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_cat_sentiment, use_container_width=True)
    
    with tab4:
        st.markdown('<h3 class="sub-header">🕸️ 기술 간 연관성 네트워크</h3>', unsafe_allow_html=True)
        
        # 네트워크 시각화
        network_fig = create_network_visualization(analyzer)
        st.plotly_chart(network_fig, use_container_width=True)
        
        # 기술 연관성 인사이트
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔗 기술 분야 연관성")
            st.markdown("""
            **주요 연결점:**
            - **AI/ML**이 모든 분야의 중심
            - **자율시스템**은 AI/ML과 강하게 연결
            - **보안/해킹**은 모든 기술 분야에 영향
            - **법률/규제**는 AI와 보안 분야를 중점 다룸
            """)
        
        with col2:
            st.markdown("### 📊 기술 융합 트렌드")
            
            # 기술 조합 분석
            tech_combinations = defaultdict(int)
            for article in filtered_articles:
                title_content = f"{article['title']} {article['content']}".lower()
                mentioned_techs = []
                
                for category, info in TECH_CATEGORIES.items():
                    if any(keyword.lower() in title_content for keyword in info['keywords']):
                        mentioned_techs.append(category)
                
                if len(mentioned_techs) > 1:
                    combo = " + ".join(sorted(mentioned_techs))
                    tech_combinations[combo] += 1
            
            if tech_combinations:
                st.markdown("**자주 언급되는 기술 조합:**")
                for combo, count in sorted(tech_combinations.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(f"- {combo}: {count}회")
            else:
                st.markdown("기술 융합 사례를 더 분석하려면 더 많은 데이터가 필요합니다.")
    
    with tab5:
        st.markdown('<h3 class="sub-header">📰 최신 기술 이슈 피드</h3>', unsafe_allow_html=True)
        
        # 최신 기사 목록
        latest_articles = sorted(filtered_articles, key=lambda x: x['date'], reverse=True)[:10]
        
        for i, article in enumerate(latest_articles):
            with st.expander(f"📄 {article['title']}", expanded=(i < 2)):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(article['content'])
                    st.markdown(f"🔗 [원문 보기]({article['url']})")
                
                with col2:
                    # 카테고리 배지
                    color = TECH_CATEGORIES.get(article['category'], {}).get('color', '#888888')
                    st.markdown(
                        f'<span class="tech-category" style="background-color: {color}">{article["category"]}</span>',
                        unsafe_allow_html=True
                    )
                    
                    flag = COUNTRIES.get(article['country'], "🏳️")
                    st.write(f"**🌍 국가:** {flag} {article['country']}")
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
            <p>📊 <strong>데이터 현황:</strong> 샘플 데이터 | 
            🔄 <strong>마지막 업데이트:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            📈 <strong>분석 기사 수:</strong> {len(filtered_articles):,}개</p>
        </div>''',
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
            <h4>🔬 국제 시사 탐구 동아리 - 글로벌 기술 이슈 분석 프로젝트</h4>
            <p>Streamlit Cloud 독립형 버전 | 데이터 기반 의사결정과 미래 기술 트렌드 예측</p>
            <p><small>🎉 성공적으로 배포되었습니다! Made with ❤️ using Streamlit & Python</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
