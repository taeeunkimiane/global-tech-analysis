# advanced_analytics.py
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from typing import List, Dict, Tuple
import logging
from collections import Counter, defaultdict
import re

logger = logging.getLogger(__name__)

class TechTrendAnalyzer:
    def __init__(self, articles: List[Dict]):
        self.articles = articles
        self.df = pd.DataFrame(articles)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # 전처리
        self._preprocess_data()
        
    def _preprocess_data(self):
        """데이터 전처리"""
        # 텍스트 결합
        self.df['full_text'] = self.df['title'] + ' ' + self.df['content']
        
        # 텍스트 정리
        self.df['clean_text'] = self.df['full_text'].apply(self._clean_text)
        
        # 날짜 관련 피처 추가
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 특수문자 제거 (단, 한글, 영문, 숫자는 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def get_trending_topics(self, n_topics: int = 5) -> Dict:
        """토픽 모델링을 통한 트렌딩 주제 분석"""
        
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

    def analyze_country_tech_focus(self) -> Dict:
        """국가별 기술 집중도 분석"""
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

    def detect_emerging_trends(self, window_days: int = 7) -> Dict:
        """신흥 트렌드 감지"""
        recent_date = self.df['date'].max()
        cutoff_date = recent_date - timedelta(days=window_days)
        
        recent_articles = self.df[self.df['date'] >= cutoff_date]
        older_articles = self.df[self.df['date'] < cutoff_date]
        
        # 최근 기간과 이전 기간의 키워드 빈도 비교
        recent_keywords = self._extract_keywords(recent_articles['clean_text'])
        older_keywords = self._extract_keywords(older_articles['clean_text'])
        
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

    def _extract_keywords(self, texts: pd.Series, n_keywords: int = 50) -> Dict[str, int]:
        """텍스트에서 키워드 추출"""
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

    def sentiment_trend_analysis(self) -> Dict:
        """감정 트렌드 분석"""
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

    def create_technology_network(self) -> nx.Graph:
        """기술 간 연관성 네트워크 생성"""
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

    def generate_insights_report(self) -> str:
        """인사이트 보고서 생성"""
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
            report.append(f"- {country}: {count}개 ({percentage:.1f}%)")
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
                report.append(f"- {country}: {top_tech} ({focus_rate:.1%} 집중)")
        
        return "\n".join(report)

class AdvancedVisualizer:
    def __init__(self, analyzer: TechTrendAnalyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
    
    def create_network_visualization(self) -> go.Figure:
        """기술 네트워크 시각화"""
        G = self.analyzer.create_technology_network()
        
        if len(G.nodes()) == 0:
            # 빈 그래프인 경우
            fig = go.Figure()
            fig.add_annotation(
                text="충분한 기술 연관성 데이터가 없습니다.",
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
    
    def create_sentiment_timeline(self) -> go.Figure:
        """감정 트렌드 타임라인"""
        sentiment_data = self.analyzer.sentiment_trend_analysis()
        daily_sentiment = sentiment_data['daily_trend']
        
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
    
    def create_topic_wordcloud_data(self) -> Dict:
        """워드클라우드용 토픽 데이터 생성"""
        trending_topics = self.analyzer.get_trending_topics()
        
        wordcloud_data = {}
        for topic_name, topic_info in trending_topics['topics'].items():
            words = topic_info['words']
            weights = topic_info['weight']
            
            # 단어-가중치 딕셔너리 생성
            word_freq = dict(zip(words, weights))
            wordcloud_data[topic_name] = word_freq
        
        return wordcloud_data

# 메인 실행부
if __name__ == "__main__":
    # 예시 데이터로 테스트
    sample_articles = [
        {
            "title": "스핀트로닉스 기술의 새로운 돌파구",
            "content": "연구진이 스핀트로닉스 기술을 이용해 기존보다 100배 빠른 연산속도를 달성했다.",
            "country": "미국",
            "category": "하드웨어 혁신",
            "date": "2025-07-01",
            "sentiment": 0.8,
            "source": "Tech Review"
        },
        {
            "title": "AI 규제 법안 통과, 업계 반발",
            "content": "새로운 AI 규제 법안이 통과되면서 기술 업계에서 강한 반발이 일고 있다.",
            "country": "독일",
            "category": "법률/규제", 
            "date": "2025-06-30",
            "sentiment": -0.3,
            "source": "Euro News"
        }
    ]
    
    analyzer = TechTrendAnalyzer(sample_articles)
    visualizer = AdvancedVisualizer(analyzer)
    
    # 분석 결과
    print(analyzer.generate_insights_report())
