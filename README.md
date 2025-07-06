# global-tech-analysis
# 🌐 글로벌 기술 이슈 분석 시스템

## 📋 프로젝트 개요

**국제 시사 탐구 동아리**에서 진행한 글로벌 과학기술 이슈와 AI 기술 발전의 상관관계를 주제로 한 심화 탐구 프로젝트입니다. 

각국 신문 기사를 주제별로 스크랩하고, 국가별·이슈별로 분류한 뒤, 웹사이트를 통해 시각적으로 분석하는 시스템을 구축했습니다. 단순히 '어디서 무슨 일이 일어나는가'를 보는 데 그치지 않고, **어떤 기술이 어떤 사회적, 정치적 배경에서 등장하는가**에 주목하여 AI·로봇·보안·디지털 규제 분야에서 국제적으로 형성되는 패러다임을 다각도로 분석합니다.

### 🎯 프로젝트 목표

- **데이터 기반 통찰력**: 실시간 뉴스 데이터 수집 및 분석을 통한 객관적 현황 파악
- **국제적 시각**: 여러 국가의 기술 정책과 사회적 반응 비교 분석
- **트렌드 예측**: 신흥 기술 키워드와 감정 변화를 통한 미래 전망
- **시각적 인사이트**: 인터랙티브 대시보드를 통한 직관적 정보 전달

## 🏗️ 시스템 아키텍처

```
📁 프로젝트 구조
├── 📄 app.py                 # 메인 Streamlit 애플리케이션
├── 📄 news_scraper.py        # 뉴스 스크래핑 모듈
├── 📄 advanced_analytics.py  # 고급 분석 및 시각화 모듈
├── 📄 requirements.txt       # Python 패키지 의존성
├── 📄 Dockerfile            # Docker 컨테이너 설정
├── 📄 docker-compose.yml    # Docker Compose 설정
├── 📁 data/                 # 수집된 데이터 저장소
│   └── 📄 tech_news_data.json
├── 📁 config/               # 설정 파일들
├── 📁 utils/                # 유틸리티 함수들
└── 📄 README.md             # 프로젝트 문서
```

## 🚀 주요 기능

### 1. 📰 실시간 뉴스 수집
- **다중 소스 지원**: MIT Technology Review, Wired, TechCrunch, IEEE Spectrum 등
- **RSS 피드 분석**: 자동화된 기사 수집 및 분류
- **Google News 검색**: 키워드 기반 추가 기사 수집
- **중복 제거**: URL 기반 고유 기사만 저장

### 2. 🧠 지능형 분석
- **자동 분류**: AI 기반 국가 및 기술 카테고리 분류
- **감정 분석**: TextBlob을 이용한 기사 감정 점수 계산
- **토픽 모델링**: LDA 알고리즘을 통한 숨겨진 주제 발견
- **트렌드 감지**: 시계열 분석을 통한 신흥 키워드 탐지

### 3. 📊 인터랙티브 대시보드
- **실시간 메트릭**: 기사 수, 국가 수, 감정 점수 등 핵심 지표
- **다층 분석**: 개요, 국가별, 기술 트렌드, 연관성, 심화 분석 탭
- **동적 필터링**: 날짜, 국가, 기술 카테고리별 맞춤 분석
- **데이터 내보내기**: CSV 형태로 분석 결과 다운로드

### 4. 🔍 고급 분석 기능
- **네트워크 분석**: 기술 간 연관성 시각화
- **경쟁력 매트릭스**: 국가별 기술 분야 점유율
- **생명주기 분석**: 이슈의 지속 기간과 강도 분석
- **키워드 동시출현**: 용어 간 관계 패턴 발견

## 🛠️ 기술 스택

### Frontend & Dashboard
- **Streamlit** 1.28.1 - 웹 애플리케이션 프레임워크
- **Plotly** 5.17.0 - 인터랙티브 시각화
- **HTML/CSS** - 커스텀 스타일링

### Data Processing & Analytics
- **Pandas** 2.1.4 - 데이터 조작 및 분석
- **NumPy** 1.24.3 - 수치 계산
- **Scikit-learn** 1.3.2 - 머신러닝 알고리즘
- **NetworkX** 3.2.1 - 네트워크 분석

### Web Scraping & NLP
- **BeautifulSoup** 4.12.2 - HTML 파싱
- **Requests** 2.31.0 - HTTP 클라이언트
- **TextBlob** 0.17.1 - 자연어 처리
- **Feedparser** 6.0.10 - RSS 피드 처리

### Deployment
- **Docker** - 컨테이너화
- **Streamlit Cloud** - 클라우드 배포
- **GitHub Actions** - CI/CD 파이프라인

## 📦 설치 및 실행

### 로컬 환경에서 실행

1. **저장소 클론**
```bash
git clone https://github.com/your-username/global-tech-analysis.git
cd global-tech-analysis
```

2. **가상환경 생성 및 활성화**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **패키지 설치**
```bash
pip install -r requirements.txt
```

4. **애플리케이션 실행**
```bash
streamlit run app.py
```

5. **브라우저 접속**
```
http://localhost:8501
```

### Docker를 이용한 실행

1. **Docker 이미지 빌드**
```bash
docker build -t global-tech-analysis .
```

2. **컨테이너 실행**
```bash
docker run -p 8501:8501 global-tech-analysis
```

### Docker Compose 사용

```bash
docker-compose up -d
```

## 📊 데이터 소스

### 🌍 뉴스 소스
- **미국**: MIT Technology Review, Wired, TechCrunch, IEEE Spectrum
- **유럽**: The Next Web, ZDNet Europe
- **아시아**: Nikkei Asia, Tech in Asia
- **검색**: Google News API (키워드 기반)

### 🔬 분석 대상 기술
- **하드웨어 혁신**: 스핀트로닉스, 마그노닉스, 양자컴퓨팅, 반도체
- **AI/ML**: 인공지능, 머신러닝, 딥러닝, 베이지안 AI
- **보안/해킹**: 사이버보안, 데이터보안, 적대적 공격
- **법률/규제**: AI 법안, 데이터 보호, 거버넌스
- **자율시스템**: 자율주행, 로봇, 드론, IoT

### 🌏 분석 대상 국가
미국, 중국, 일본, 독일, 영국, 프랑스, 한국, 이스라엘, 싱가포르, 캐나다

## 📈 분석 방법론

### 1. 데이터 수집
```python
# 뉴스 스크래퍼 사용 예시
from news_scraper import GlobalTechNewsScraper

scraper = GlobalTechNewsScraper()
articles = scraper.scrape_all_sources()
scraper.save_to_json(articles)
```

### 2. 분석 실행
```python
# 고급 분석 모듈 사용 예시
from advanced_analytics import TechTrendAnalyzer

analyzer = TechTrendAnalyzer(articles)
insights = analyzer.generate_insights_report()
trending = analyzer.detect_emerging_trends()
```

### 3. 시각화
```python
# 시각화 모듈 사용 예시
from advanced_analytics import AdvancedVisualizer

visualizer = AdvancedVisualizer(analyzer)
network_fig = visualizer.create_network_visualization()
sentiment_fig = visualizer.create_sentiment_timeline()
```

## 🔍 핵심 인사이트

### 📊 발견된 패턴
1. **기술 패권 경쟁**: 미-중 간 AI 및 반도체 분야 집중 투자
2. **규제 선도국**: EU의 AI 법안이 글로벌 표준으로 확산
3. **신흥 기술 트렌드**: 스핀트로닉스, 비모수 베이지안 AI 등 차세대 기술 부상
4. **감정 변화**: 기술 발전에 대한 기대와 보안/윤리 우려의 공존

### 🔮 예측 모델
- **트렌드 점수**: 최근 7일 vs 이전 기간 키워드 빈도 비교
- **감정 궤적**: 시간별 여론 변화 패턴 분석
- **기술 생명주기**: 이슈의 지속성과 강도 예측

## 🚀 배포 및 운영

### Streamlit Cloud 배포
1. GitHub 저장소 연결
2. `streamlit run app.py` 명령어 설정
3. 환경 변수 설정 (API 키 등)
4. 자동 배포 및 업데이트

### 모니터링
- **데이터 품질**: 수집된 기사 수, 분류 정확도
- **시스템 성능**: 응답 시간, 메모리 사용량
- **사용자 활동**: 페이지 뷰, 필터 사용 패턴

## 🔧 개발 로드맵

### Phase 1: 기본 시스템 구축 ✅
- [x] 뉴스 스크래핑 시스템
- [x] 기본 분석 및 시각화
- [x] Streamlit 대시보드

### Phase 2: 고급 분석 기능 ✅
- [x] 토픽 모델링
- [x] 네트워크 분석
- [x] 감정 트렌드 분석

### Phase 3: 확장 기능 🚧
- [ ] 실시간 알림 시스템
- [ ] API 서비스 제공
- [ ] 모바일 앱 개발
- [ ] 다국어 지원

### Phase 4: AI 강화 🔮
- [ ] 고급 NLP 모델 적용
- [ ] 예측 모델 정확도 향상
- [ ] 자동화된 인사이트 생성

## 👥 기여 방법

### 개발 환경 설정
1. Fork this repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request

### 코딩 컨벤션
- **Python**: PEP 8 스타일 가이드 준수
- **함수명**: snake_case 사용
- **클래스명**: PascalCase 사용
- **상수**: UPPER_CASE 사용

## 🐛 알려진 이슈

### 제한사항
- **API 제한**: 일부 뉴스 소스의 요청 빈도 제한
- **언어 처리**: 영어 기사에 최적화 (한국어 분석 정확도 개선 필요)
- **실시간성**: RSS 피드 업데이트 주기에 의존

### 해결책
- **캐싱 시스템**: 중복 요청 방지 및 응답 속도 개선
- **다국어 NLP**: 언어별 특화 모델 적용 예정
- **Webhook 연동**: 실시간 업데이트 시스템 구축 예정

## 📚 참고 자료

### 학술 논문
- "Spintronics: A Spin-Based Electronics Vision for the Future" (Science, 2001)
- "Non-parametric Bayesian Methods for Machine Learning" (JMLR, 2011)
- "Global AI Governance: A Multilateral Approach" (Nature Machine Intelligence, 2023)

### 기술 문서
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### 데이터 소스
- [MIT Technology Review RSS](https://www.technologyreview.com/feed/)
- [IEEE Spectrum Tech Talk](https://spectrum.ieee.org/rss/blog/tech-talk)
- [Google News API](https://developers.google.com/news)

## 📞 연락처

**프로젝트 리더**: [Your Name]  
**이메일**: your.email@example.com  
**GitHub**: [@your-username](https://github.com/your-username)  
**동아리**: 국제 시사 탐구 동아리

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **데이터 제공**: 각종 뉴스 소스 및 RSS 피드 제공자들
- **오픈소스 커뮤니티**: Streamlit, Plotly, scikit-learn 개발팀
- **동아리 구성원들**: 프로젝트 기획 및 피드백 제공

---

<div align="center">

**🌐 글로벌 기술 이슈 분석 시스템**  
*데이터로 읽는 미래 기술의 흐름*

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com/)

</div>
