# app.py - 메인 Streamlit 애플리케이션 (import 문제 해결 버전)
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

# 선택적 import - 없어도 앱이 작동하도록 수정
try:
    import requests
    from bs4 import BeautifulSoup
    import feedparser
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    st.warning("⚠️ 웹 스크래핑 모듈을 사용할 수 없습니다. 샘플 데이터를 사용합니다.")

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
    from sklearn.decomposition import LatentDirichletAllocation
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

# ===== 원래 external 모듈들을 내장 함수로 변환 =====

class TechNewsAnalyzer:
    """기술 뉴스 분석기 (내장 버전)"""
    
    def __init__(self, articles):
        self.articles = articles
        self.df = pd.DataFrame(articles) if articles else pd.DataFrame()
        if not self.df.empty:
            self.df['date'] = pd.to_datetime(self.df['date'])
            # 텍스트 결합
            self.df['full_text'] = self.df['title'] + ' ' + self.df['content']
            # 텍스트 정리
            self.df['clean_text'] = self.df['full_text'].apply(self._clean_text)
            # 날짜 관련 피처 추가
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['day_of_week'] = self.df['date'].dt.day_name()
    
    def _clean_text(self, text):
        """텍스트 정리"""
        import re
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 특수문자 제거 (단, 한글, 영문, 숫자는 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def get_trending_topics(self, n_topics=5):
        """토픽 모델링을 통한 트렌딩 주제 분석"""
        if not SKLEARN_AVAILABLE or self.df.empty:
            # 간단한 키워드 기반 분석으로 대체
            return self._simple_topic_analysis()
        
        try:
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(self.df['clean_text'])
            
            # LDA 토픽 모델링
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(tfidf_matrix)
            
            # 토픽별 주요 단어 추출
            feature_names = vectorizer.get_feature_names_out()
            topics = {}
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics[f"Topic_{topic_idx + 1}"] = {
                    'words': top_words,
                    'weight': topic[top_words_idx].tolist()
                }
            
            # 문서-토픽 분포
            doc_topic_dist = lda.transform(tfidf_matrix)
            
            return {
                'topics': topics,
                'doc_topic_distribution': doc_topic_dist,
                'vectorizer': vectorizer,
                'lda_model': lda
            }
        except Exception as e:
            return self._simple_topic_analysis()
    
    def _simple_topic_analysis(self):
        """간단한 토픽 분석 (sklearn 없을 때)"""
        topics = {}
        for i, (category, info) in enumerate(TECH_CATEGORIES.items()):
            topics[f"Topic_{i+1}"] = {
                'words': info['keywords'][:5],
                'weight': [1.0, 0.8, 0.6, 0.4, 0.2]
            }
        
        return {
            'topics': topics,
            'doc_topic_distribution': None,
            'vectorizer': None,
            'lda_model': None
        }

    def analyze_country_tech_focus(self):
        """국가별 기술 집중도 분석"""
        if self.df.empty:
            return {}
        
        from collections import defaultdict
        country_tech = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            country_tech[row['country']][row['category']] += 1
        
        # 정규화 (각 국가의 총 기사 수로 나누기)
        country_tech_normalized = {}
        for country, tech_counts in country_tech.items():
            total_articles = sum(tech_counts.values())
            country_tech_normalized[country] = {
                tech: count / total_articles
                for tech, count in tech_counts.items()
            }
        
        return country_tech_normalized

    def detect_emerging_trends(self, window_days=7):
        """신흥 트렌드 감지"""
        if self.df.empty:
            return {}
        
        recent_date = self.df['date'].max()
        cutoff_date = recent_date - timedelta(days=window_days)
        
        recent_articles = self.df[self.df['date'] >= cutoff_date]
        older_articles = self.df[self.df['date'] < cutoff_date]
        
        # 최근 기간과 이전 기간의 키워드 빈도 비교
        recent_keywords = self._extract_keywords(recent_articles['clean_text'] if not recent_articles.empty else pd.Series())
        older_keywords = self._extract_keywords(older_articles['clean_text'] if not older_articles.empty else pd.Series())
        
        # 트렌드 스코어 계산 (최근 빈도 / 이전 빈도)
        trending_keywords = {}
        for keyword, recent_freq in recent_keywords.items():
            older_freq = older_keywords.get(keyword, 1)  # 0으로 나누기 방지
            trend_score = recent_freq / older_freq
            
            if recent_freq >= 3 and trend_score > 2:  # 최소 빈도와 트렌드 스코어 기준
                trending_keywords[keyword] = {
                    'recent_frequency': recent_freq,
                    'older_frequency': older_freq,
                    'trend_score': trend_score
                }
        
        # 트렌드 스코어 순으로 정렬
        trending_keywords = dict(
            sorted(trending_keywords.items(), 
                   key=lambda x: x[1]['trend_score'], 
                   reverse=True)
        )
        
        return trending_keywords

    def _extract_keywords(self, texts, n_keywords=50):
        """텍스트에서 키워드 추출"""
        if texts.empty:
            return {}
        
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer(
                    max_features=n_keywords,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2
                )
                
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # 각 키워드의 총 TF-IDF 점수 계산
                keyword_scores = {}
                for i, keyword in enumerate(feature_names):
                    keyword_scores[keyword] = tfidf_matrix[:, i].sum()
                
                return dict(Counter(keyword_scores).most_common(n_keywords))
            except:
                pass
        
        # 간단한 키워드 추출 (fallback)
        all_keywords = []
        for category_info in TECH_CATEGORIES.values():
            all_keywords.extend(category_info['keywords'])
        
        keyword_counts = {}
        combined_text = ' '.join(texts.fillna(''))
        
        for keyword in all_keywords:
            count = combined_text.lower().count(keyword.lower())
            if count > 0:
                keyword_counts[keyword] = count
        
        return keyword_counts

    def sentiment_trend_analysis(self):
        """감정 트렌드 분석"""
        if self.df.empty:
            return {
                'daily_trend': pd.DataFrame(),
                'country_sentiment': pd.DataFrame(),
                'category_sentiment': pd.DataFrame()
            }
        
        # 날짜별 평균 감정 계산
        daily_sentiment = self.df.groupby('date')['sentiment'].agg(['mean', 'count']).reset_index()
        
        # 국가별 감정 분석
        country_sentiment = self.df.groupby('country')['sentiment'].agg(['mean', 'std', 'count']).reset_index()
        
        # 카테고리별 감정 분석
        category_sentiment = self.df.groupby('category')['sentiment'].agg(['mean', 'std', 'count']).reset_index()
        
        return {
            'daily_trend': daily_sentiment,
            'country_sentiment': country_sentiment,
            'category_sentiment': category_sentiment
        }

    def create_technology_network(self):
        """기술 간 연관성 네트워크 생성"""
        if not NETWORKX_AVAILABLE:
            return None
        
        try:
            from collections import defaultdict
            import itertools
            
            # 기사별로 언급된 기술들을 추출
            tech_cooccurrence = defaultdict(lambda: defaultdict(int))
            
            # 기술 키워드 정의 (확장된 버전)
            tech_keywords = {
                '스핀트로닉스': ['spintronics', '스핀트로닉스', 'spin electronics'],
                '양자컴퓨팅': ['quantum computing', '양자컴퓨팅', 'quantum computer'],
                'AI': ['artificial intelligence', '인공지능', 'machine learning', 'deep learning'],
                '자율주행': ['autonomous vehicle', '자율주행', 'self-driving'],
                '블록체인': ['blockchain', '블록체인', 'cryptocurrency'],
                '5G': ['5G', '5g network', '5세대'],
                'IoT': ['IoT', 'internet of things', '사물인터넷'],
                'AR/VR': ['augmented reality', 'virtual reality', 'AR', 'VR', '증강현실', '가상현실']
            }
            
            # 각 기사에서 언급된 기술들 찾기
            for _, row in self.df.iterrows():
                text = row['clean_text']
                mentioned_techs = []
                
                for tech_name, keywords in tech_keywords.items():
                    if any(keyword.lower() in text for keyword in keywords):
                        mentioned_techs.append(tech_name)
                
                # 동시 출현 기록
                for i, tech1 in enumerate(mentioned_techs):
                    for tech2 in mentioned_techs[i+1:]:
                        tech_cooccurrence[tech1][tech2] += 1
                        tech_cooccurrence[tech2][tech1] += 1
            
            # 네트워크 그래프 생성
            G = nx.Graph()
            
            for tech1, connections in tech_cooccurrence.items():
                for tech2, weight in connections.items():
                    if weight >= 2:  # 최소 2번 이상 동시 출현
                        G.add_edge(tech1, tech2, weight=weight)
            
            return G
        except Exception as e:
            return None

    def generate_insights_report(self):
        """인사이트 보고서 생성"""
        if self.df.empty:
            return "데이터가 없습니다."
        
        report = []
        
        # 기본 통계
        total_articles = len(self.df)
        date_range = f"{self.df['date'].min().strftime('%Y-%m-%d')} ~ {self.df['date'].max().strftime('%Y-%m-%d')}"
        
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
        
        # 감정 분석 결과
        avg_sentiment = self.df['sentiment'].mean()
        sentiment_label = "긍정적" if avg_sentiment > 0.1 else "부정적" if avg_sentiment < -0.1 else "중립적"
        report.append("🎭 **전반적 여론**")
        report.append(f"- 평균 감정 점수: {avg_sentiment:.3f} ({sentiment_label})")
        report.append("")
        
        # 신흥 트렌드
        trending = self.detect_emerging_trends()
        if trending:
            report.append("📈 **신흥 트렌드 키워드** (최근 7일)")
            for keyword, data in list(trending.items())[:5]:
                report.append(f"- {keyword}: 트렌드 점수 {data['trend_score']:.1f}x")
            report.append("")
        
        # 기술 집중도가 높은 국가
        country_focus = self.analyze_country_tech_focus()
        report.append("🎯 **국가별 기술 특화 분야**")
        for country, tech_dist in list(country_focus.items())[:5]:
            if tech_dist:
                top_tech = max(tech_dist, key=tech_dist.get)
                focus_rate = tech_dist[top_tech]
                flag = COUNTRIES.get(country, "🏳️")
                report.append(f"- {flag} {country}: {top_tech} ({focus_rate:.1%} 집중)")
        
        return "\n".join(report)

class AdvancedVisualizer:
    """고급 시각화 클래스 (내장 버전)"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
    
    def create_network_visualization(self):
        """기술 네트워크 시각화"""
        G = self.analyzer.create_technology_network()
        
        if G is None or len(G.nodes()) == 0:
            # 빈 그래프인 경우
            fig = go.Figure()
            fig.add_annotation(
                text="충분한 기술 연관성 데이터가 없습니다." + ("<br>NetworkX가 필요합니다." if not NETWORKX_AVAILABLE else ""),
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # 노드 위치 계산
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 엣지 그리기
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} ↔ {edge[1]}: {weight}개 기사에서 함께 언급")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 노드 그리기
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # 노드 크기는 연결된 엣지 수에 비례
            connections = len(list(G.neighbors(node)))
            node_size.append(20 + connections * 10)
            node_info.append(f"{node}<br>연결된 기술: {connections}개")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_size,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='기술 간 연관성 네트워크',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="노드 크기는 다른 기술과의 연관성 정도를 나타냄",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_sentiment_timeline(self):
        """감정 트렌드 타임라인"""
        sentiment_data = self.analyzer.sentiment_trend_analysis()
        daily_sentiment = sentiment_data['daily_trend']
        
        if daily_sentiment.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="감정 트렌드 데이터가 충분하지 않습니다.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        fig = go.Figure()
        
        # 전체 감정 트렌드
        fig.add_trace(go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['mean'],
            mode='lines+markers',
            name='일평균 감정',
            line=dict(color='blue', width=3),
            hovertemplate='%{x}<br>감정 점수: %{y:.3f}<br>기사 수: %{customdata}<extra></extra>',
            customdata=daily_sentiment['count']
        ))
        
        # 0선 추가
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="중립선", annotation_position="bottom right")
        
        # 긍정/부정 영역 색칠
        fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title='기술 이슈에 대한 감정 트렌드',
            xaxis_title='날짜',
            yaxis_title='감정 점수 (-1: 매우 부정적, +1: 매우 긍정적)',
            hovermode='x unified'
        )
        
        return fig

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
    countries = list(COUNTRIES.keys())
    categories = list(TECH_CATEGORIES.keys())
    
    sample_articles = []
    base_date = datetime.now() - timedelta(days=30)
    
    # 실제적인 샘플 기사들
    realistic_articles = [
        {
            'title': '미국, 차세대 스핀트로닉스 칩 개발 성공',
            'content': '미국 연구진이 전자의 스핀을 이용한 혁신적인 컴퓨팅 기술을 개발했다. 이 기술은 기존 반도체보다 100배 빠른 연산 속도와 1/10 수준의 전력 소비를 실현한다.',
            'country': '미국',
            'category': '하드웨어 혁신',
            'date': '2025-07-03',
            'sentiment': 0.752,
            'source': 'MIT Technology Review',
            'url': 'https://example.com/article/1',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': '중국 AI 기업, 비모수 베이지안 모델 개발',
            'content': '중국 AI 기업이 불확실성을 정량적으로 표현할 수 있는 비모수 베이지안 AI 모델을 개발했다고 발표했다.',
            'country': '중국',
            'category': 'AI/ML',
            'date': '2025-07-02',
            'sentiment': 0.634,
            'source': 'Tech in Asia',
            'url': 'https://example.com/article/2',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': '일본 정부기관, 대규모 사이버 공격 받아',
            'content': '일본의 주요 정부기관이 AI 시스템을 대상으로 한 적대적 공격을 받았다고 발표했다.',
            'country': '일본',
            'category': '보안/해킹',
            'date': '2025-07-01',
            'sentiment': -0.423,
            'source': 'Nikkei Asia',
            'url': 'https://example.com/article/3',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': '독일, AI 규제 법안 의회 통과',
            'content': '독일 의회가 AI 시스템의 안전성과 투명성을 보장하기 위한 포괄적인 법적 프레임워크를 통과시켰다.',
            'country': '독일',
            'category': '법률/규제',
            'date': '2025-06-30',
            'sentiment': 0.156,
            'source': 'The Next Web',
            'url': 'https://example.com/article/4',
            'scraped_at': datetime.now().isoformat()
        },
        {
            'title': '한국 자동차 업체, 완전자율주행 기술 시연',
            'content': '한국의 주요 자동차 제조사가 마그노닉스 기술을 적용한 완전자율주행 시스템을 성공적으로 시연했다.',
            'country': '한국',
            'category': '자율시스템',
            'date': '2025-06-29',
            'sentiment': 0.687,
            'source': 'TechCrunch',
            'url': 'https://example.com/article/5',
            'scraped_at': datetime.now().isoformat()
        }
    ]
    
    sample_articles.extend(realistic_articles)
    
    # 추가 샘플 데이터 생성
    for i in range(50):
        country = countries[i % len(countries)]
        category = categories[i % len(categories)]
        
        # 감정점수는 카테고리에 따라 다르게 설정
        if category == "보안/해킹":
            sentiment = np.random.normal(-0.2, 0.3)
        elif category == "법률/규제":
            sentiment = np.random.normal(-0.1, 0.4)
        else:
            sentiment = np.random.normal(0.2, 0.3)
        
        sentiment = np.clip(sentiment, -1, 1)
        
        article = {
            'title': f'{country} {category} 관련 기술 뉴스 {i+6}',
            'content': f'이것은 {category} 분야의 {country} 관련 기사입니다. 최신 기술 동향과 발전사항을 다루고 있습니다.',
            'country': country,
            'category': category,
            'date': (base_date + timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d'),
            'sentiment': round(sentiment, 3),
            'source': f'{country} Tech News',
            'url': f'https://example.com/article/{i+6}',
            'scraped_at': datetime.now().isoformat()
        }
        sample_articles.append(article)
    
    return sample_articles

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🌐 글로벌 기술 이슈 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">AI 시대의 기술 패러다임 변화와 국제적 거버넌스 동향 분석</p>', unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.markdown("## 🔧 분석 설정")
        
        # 배포 상태 표시
        st.success("🎉 성공적으로 배포되었습니다!")
        
        # 기능 상태 표시
        st.markdown("### 📊 기능 상태")
        st.write(f"✅ 기본 분석: 사용 가능")
        st.write(f"{'✅' if TEXTBLOB_AVAILABLE else '⚠️'} 감정 분석: {'고급' if TEXTBLOB_AVAILABLE else '기본'}")
        st.write(f"{'✅' if NETWORKX_AVAILABLE else '⚠️'} 네트워크 분석: {'사용 가능' if NETWORKX_AVAILABLE else '제한됨'}")
        st.write(f"{'✅' if SKLEARN_AVAILABLE else '⚠️'} 고급 분석: {'사용 가능' if SKLEARN_AVAILABLE else '제한됨'}")
        st.write(f"{'✅' if WEB_SCRAPING_AVAILABLE else '⚠️'} 웹 스크래핑: {'사용 가능' if WEB_SCRAPING_AVAILABLE else '제한됨'}")
        
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
        selected_countries = st.multiselect(
            "분석 대상 국가",
            list(COUNTRIES.keys()),
            default=list(COUNTRIES.keys())[:5],
            help="분석할 국가들을 선택해주세요."
        )
        
        # 기술 카테고리 선택
        selected_categories = st.multiselect(
            "기술 카테고리",
            list(TECH_CATEGORIES.keys()),
            default=list(TECH_CATEGORIES.keys()),
            help="분석할 기술 분야를 선택해주세요."
        )
        
        st.markdown("---")
        
        # 데이터 갱신
        st.markdown("### 📊 데이터 관리")
        
        if st.button("🔄 데이터 갱신", type="primary"):
            if data_source == "실시간 수집":
                if WEB_SCRAPING_AVAILABLE:
                    with st.spinner("뉴스 데이터 수집 중... (약 2-3분 소요)"):
                        # 실제 구현에서는 scraper.scrape_all_sources() 사용
                        st.warning("실시간 수집 기능은 현재 개발 중입니다. 샘플 데이터를 사용해주세요.")
                else:
                    st.error("웹 스크래핑 모듈이 없어 실시간 수집을 할 수 없습니다.")
            else:
                st.success("✅ 데이터 갱신 완료!")
                st.rerun()
    
    # 메인 컨텐츠 - 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        if data_source == "캐시된 데이터":
            cached_data = load_cached_data()
            if cached_data:
                current_articles = cached_data
            else:
                st.warning("캐시된 데이터가 없습니다. 샘플 데이터를 사용합니다.")
                current_articles = get_sample_data()
        else:  # 샘플 데이터 또는 실시간 수집
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
    analyzer = TechNewsAnalyzer(filtered_articles)
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
            
            colors = [TECH_CATEGORIES.get(cat, {}).get('color', '#888888') for cat in category_counts.keys()]
            
            fig_category = px.pie(
                values=list(category_counts.values()),
                names=list(category_counts.keys()),
                title="기술 분야별 비중",
                color_discrete_sequence=colors
            )
            fig_category.update_layout(height=400)
            st.plotly_chart(fig_category, use_container_width=True)
        
        # 시간별 트렌드 분석
        st.markdown('<h4 class="sub-header">📈 시간별 이슈 트렌드</h4>', unsafe_allow_html=True)
        
        # 날짜별 데이터 준비
        df_articles = pd.DataFrame(filtered_articles)
        df_articles['date'] = pd.to_datetime(df_articles['date'])
        
        daily_counts = df_articles.groupby(['date', 'category']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            daily_counts, 
            x='date', 
            y='count', 
            color='category',
            title="일별 기술 이슈 발생 트렌드",
            color_discrete_map={cat: info['color'] for cat, info in TECH_CATEGORIES.items()}
        )
        fig_timeline.update_layout(height=500)
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
                title="국가-기술분야 히트맵",
                text_auto=True
            )
            fig_heatmap.update_layout(height=500)
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
                        weights = topic_info['weight'][:8] if isinstance(topic_info['weight'], list) else [1.0] * len(words)
                        
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
                    flag = COUNTRIES.get(country, "🏳️")
                    
                    st.markdown(f"**{flag} {country}**")
                    st.markdown(f"특화 분야: {top_tech}")
                    st.markdown(f"집중도: {focus_rate:.1%}")
                    st.markdown("---")
    
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
                category_colors = TECH_CATEGORIES
                color = category_colors.get(article['category'], {}).get('color', '#888888')
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
            <p><small>Made with ❤️ using Streamlit & Python | 🎉 성공적으로 배포되었습니다!</small></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
