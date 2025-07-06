# 🚀 빠른 시작 가이드

## 📋 목차
1. [사전 요구사항](#사전-요구사항)
2. [빠른 설치 및 실행](#빠른-설치-및-실행)
3. [Docker로 실행](#docker로-실행)
4. [개발 환경 설정](#개발-환경-설정)
5. [문제 해결](#문제-해결)

---

## 🛠️ 사전 요구사항

### 필수 요구사항
- **Python 3.9+** ([다운로드](https://www.python.org/downloads/))
- **Git** ([다운로드](https://git-scm.com/downloads))

### Docker 사용 시 (권장)
- **Docker** ([다운로드](https://docs.docker.com/get-docker/))
- **Docker Compose** ([설치 가이드](https://docs.docker.com/compose/install/))

---

## ⚡ 빠른 설치 및 실행

### 1단계: 저장소 클론
```bash
git clone https://github.com/your-username/global-tech-analysis.git
cd global-tech-analysis
```

### 2단계: 자동 설정 실행
```bash
# Linux/macOS
chmod +x setup.sh
./setup.sh local

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3단계: 애플리케이션 실행
```bash
# 가상환경 활성화 (이미 활성화된 경우 생략)
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows

# Streamlit 앱 실행
streamlit run app.py
```

### 4단계: 브라우저 접속
웹 브라우저에서 `http://localhost:8501` 접속

---

## 🐳 Docker로 실행 (가장 간단한 방법)

### 빠른 실행
```bash
# 저장소 클론
git clone https://github.com/your-username/global-tech-analysis.git
cd global-tech-analysis

# Docker Compose로 모든 서비스 시작
docker-compose up -d

# 로그 확인 (선택사항)
docker-compose logs -f
```

### 접속 주소
- **메인 애플리케이션**: http://localhost:8501
- **Nginx (프록시)**: http://localhost:80

### 서비스 중지
```bash
docker-compose down
```

---

## 💻 개발 환경 설정

### 상세 개발 환경 설정
```bash
# 1. 프로젝트 클론
git clone https://github.com/your-username/global-tech-analysis.git
cd global-tech-analysis

# 2. Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows

# 3. 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 4. 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 필요한 API 키 설정

# 5. 디렉토리 구조 생성
mkdir -p data config logs cache

# 6. 초기 데이터 수집 (선택사항)
python data_collector.py collect

# 7. 테스트 실행
pytest tests/ -v

# 8. 애플리케이션 실행
streamlit run app.py
```

### 개발 도구 설치 (선택사항)
```bash
pip install black flake8 pytest-cov
```

---

## 📊 샘플 데이터로 시작하기

처음 실행 시 실제 뉴스 데이터가 없다면 샘플 데이터를 사용할 수 있습니다:

1. 애플리케이션 실행 후 사이드바에서 **"샘플 데이터"** 선택
2. **"데이터 갱신"** 버튼 클릭
3. 자동으로 생성된 샘플 데이터로 분석 기능 체험

---

## 🔧 기본 사용법

### 1. 데이터 필터링
- **국가 선택**: 사이드바에서 관심 있는 국가 선택
- **기술 분야**: 분석할 기술 카테고리 선택
- **날짜 범위**: 분석 기간 설정

### 2. 대시보드 탐색
- **📊 개요 대시보드**: 전체 현황 및 트렌드
- **🌍 국가별 분석**: 국가별 기술 이슈 분석
- **🔬 기술 트렌드**: 신흥 기술 트렌드 분석
- **🕸️ 연관성 분석**: 기술 간 네트워크 분석
- **📈 심화 분석**: 고급 분석 기능

### 3. 데이터 내보내기
- 분석 결과를 CSV 또는 JSON 형태로 다운로드
- 시각화를 이미지로 저장

---

## ⚠️ 문제 해결

### 일반적인 문제들

#### 1. "ModuleNotFoundError" 오류
```bash
# 해결방법: 의존성 재설치
pip install -r requirements.txt

# 또는 특정 패키지 설치
pip install streamlit plotly pandas
```

#### 2. 포트 8501이 이미 사용 중
```bash
# 다른 포트로 실행
streamlit run app.py --server.port 8502

# 또는 기존 프로세스 종료
lsof -ti:8501 | xargs kill -9  # Linux/macOS
```

#### 3. Docker 컨테이너가 시작되지 않음
```bash
# 로그 확인
docker-compose logs

# 컨테이너 재시작
docker-compose restart

# 완전 재빌드
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

#### 4. 메모리 부족 오류
```bash
# Docker 메모리 제한 늘리기 (docker-compose.yml 수정)
deploy:
  resources:
    limits:
      memory: 2G  # 기본 1G에서 2G로 증가
```

#### 5. SSL/HTTPS 관련 오류
```bash
# 자체 서명 인증서 생성
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem \
  -subj "/C=KR/ST=Seoul/L=Seoul/O=TechAnalysis/CN=localhost"
```

### 성능 최적화

#### 메모리 사용량 줄이기
```python
# config/settings.json에서 설정 조정
{
  "cache": {
    "enabled": true,
    "ttl_hours": 12,  # 캐시 유지 시간 단축
    "max_size_mb": 200  # 캐시 크기 제한
  },
  "analysis": {
    "max_articles": 1000  # 분석할 최대 기사 수 제한
  }
}
```

#### 느린 로딩 시간 개선
1. **캐시 활성화**: 사이드바에서 "캐시된 데이터" 사용
2. **샘플 데이터 사용**: 빠른 테스트를 위해 샘플 데이터 활용
3. **필터 조정**: 분석 범위를 좁혀 처리 시간 단축

---

## 📚 추가 리소스

### 문서
- [전체 문서](README.md)
- [API 문서](docs/API.md) (향후 제공)
- [배포 가이드](docs/DEPLOYMENT.md) (향후 제공)

### 지원
- **GitHub Issues**: [이슈 신고](https://github.com/your-username/global-tech-analysis/issues)
- **토론**: [GitHub Discussions](https://github.com/your-username/global-tech-analysis/discussions)

### 커뮤니티
- **동아리 커뮤니티**: 국제 시사 탐구 동아리
- **이메일**: your.email@example.com

---

## ✅ 빠른 체크리스트

설치가 완료되었는지 확인하세요:

- [ ] Python 3.9+ 설치됨
- [ ] Git 설치됨
- [ ] 저장소 클론 완료
- [ ] 가상환경 생성 및 활성화
- [ ] 의존성 설치 완료
- [ ] 애플리케이션이 http://localhost:8501에서 실행됨
- [ ] 대시보드가 정상적으로 로드됨
- [ ] 샘플 데이터 또는 실제 데이터로 분석 가능

모든 항목이 체크되었다면 성공적으로 설치가 완료되었습니다! 🎉

---

## 🚀 다음 단계

1. **데이터 수집 설정**: API 키를 설정하여 실시간 뉴스 수집
2. **분석 탐구**: 다양한 필터와 분석 기능 탐색
3. **커스터마이징**: 설정 파일을 수정하여 개인화
4. **배포**: 프로덕션 환경에 배포 (Docker 권장)

---

<div align="center">

**🌐 글로벌 기술 이슈 분석 시스템**  
*데이터로 읽는 미래 기술의 흐름*

[⬆️ 목차로 돌아가기](#목차) | [📖 전체 문서](README.md) | [🐛 이슈 신고](https://github.com/your-username/global-tech-analysis/issues)

</div>
