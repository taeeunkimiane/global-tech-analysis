#!/bin/bash

# 글로벌 기술 이슈 분석 시스템 설정 스크립트
# 사용법: ./setup.sh [local|docker|production]

set -e  # 에러 발생 시 스크립트 중단

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수들
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 배너 출력
print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║    🌐 글로벌 기술 이슈 분석 시스템 설정                        ║
    ║                                                               ║
    ║    AI 시대의 기술 패러다임 변화와 국제적 거버넌스 동향 분석    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 시스템 요구사항 확인
check_requirements() {
    log_info "시스템 요구사항 확인 중..."
    
    # Python 확인
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3가 설치되어 있지 않습니다. Python 3.9 이상을 설치해주세요."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Python 버전: $PYTHON_VERSION"
    
    # pip 확인
    if ! command -v pip3 &> /dev/null; then
        log_error "pip가 설치되어 있지 않습니다."
        exit 1
    fi
    
    # Git 확인
    if ! command -v git &> /dev/null; then
        log_warning "Git이 설치되어 있지 않습니다. 일부 기능이 제한될 수 있습니다."
    fi
    
    log_success "시스템 요구사항 확인 완료"
}

# Docker 요구사항 확인
check_docker_requirements() {
    log_info "Docker 요구사항 확인 중..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되어 있지 않습니다."
        log_info "Docker 설치 가이드: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose가 설치되어 있지 않습니다."
        log_info "Docker Compose 설치 가이드: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Docker 서비스 상태 확인
    if ! docker info &> /dev/null; then
        log_error "Docker 서비스가 실행되고 있지 않습니다."
        log_info "Docker 서비스를 시작해주세요: sudo systemctl start docker"
        exit 1
    fi
    
    log_success "Docker 요구사항 확인 완료"
}

# 디렉토리 구조 생성
create_directories() {
    log_info "디렉토리 구조 생성 중..."
    
    directories=(
        "data"
        "config"
        "logs" 
        "cache"
        "ssl"
        "tests"
        "tests/unit"
        "tests/integration"
        "tests/performance"
        ".github/workflows"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_info "디렉토리 생성: $dir"
    done
    
    log_success "디렉토리 구조 생성 완료"
}

# 로컬 환경 설정
setup_local() {
    log_info "로컬 개발 환경 설정 중..."
    
    # 가상환경 생성
    if [ ! -d "venv" ]; then
        log_info "Python 가상환경 생성 중..."
        python3 -m venv venv
        log_success "가상환경 생성 완료"
    else
        log_info "기존 가상환경 발견"
    fi
    
    # 가상환경 활성화
    source venv/bin/activate
    log_info "가상환경 활성화 완료"
    
    # 패키지 설치
    log_info "Python 패키지 설치 중..."
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "패키지 설치 완료"
    
    # NLTK 데이터 다운로드
    log_info "NLTK 데이터 다운로드 중..."
    python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
    log_success "NLTK 데이터 다운로드 완료"
    
    # 환경 변수 파일 생성
    if [ ! -f ".env" ]; then
        log_info "환경 변수 파일 생성 중..."
        cat > .env << EOF
# 글로벌 기술 이슈 분석 시스템 환경 변수

# 데이터 수집 설정
COLLECTION_INTERVAL=21600
NEWS_API_KEY=your_news_api_key_here
GOOGLE_NEWS_API_KEY=your_google_news_api_key_here

# 캐시 설정
CACHE_TTL_HOURS=24
CACHE_MAX_SIZE_MB=500

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Streamlit 설정
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EOF
        log_success "환경 변수 파일 생성 완료"
        log_warning "필요한 API 키를 .env 파일에 설정해주세요."
    fi
    
    log_success "로컬 환경 설정 완료"
}

# Docker 환경 설정
setup_docker() {
    log_info "Docker 환경 설정 중..."
    
    # Docker 이미지 빌드
    log_info "Docker 이미지 빌드 중..."
    docker-compose build
    log_success "Docker 이미지 빌드 완료"
    
    # 서비스 시작
    log_info "Docker 서비스 시작 중..."
    docker-compose up -d
    log_success "Docker 서비스 시작 완료"
    
    # 헬스체크
    log_info "서비스 상태 확인 중..."
    sleep 30
    
    if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
        log_success "애플리케이션이 정상적으로 실행 중입니다."
        log_info "접속 URL: http://localhost:8501"
    else
        log_warning "애플리케이션 헬스체크 실패. 로그를 확인해주세요."
        log_info "로그 확인: docker-compose logs -f"
    fi
    
    log_success "Docker 환경 설정 완료"
}

# 프로덕션 환경 설정
setup_production() {
    log_info "프로덕션 환경 설정 중..."
    
    # SSL 인증서 설정
    if [ ! -f "ssl/cert.pem" ] || [ ! -f "ssl/key.pem" ]; then
        log_warning "SSL 인증서가 없습니다."
        read -p "Let's Encrypt를 사용하여 SSL 인증서를 생성하시겠습니까? (y/n): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if command -v certbot &> /dev/null; then
                read -p "도메인 이름을 입력하세요: " domain
                read -p "이메일 주소를 입력하세요: " email
                
                certbot certonly --standalone -d "$domain" --email "$email" --agree-tos --non-interactive
                
                # 인증서 복사
                cp "/etc/letsencrypt/live/$domain/fullchain.pem" ssl/cert.pem
                cp "/etc/letsencrypt/live/$domain/privkey.pem" ssl/key.pem
                
                log_success "SSL 인증서 설정 완료"
            else
                log_error "Certbot이 설치되어 있지 않습니다."
                log_info "수동으로 SSL 인증서를 ssl/ 디렉토리에 배치해주세요."
            fi
        else
            log_info "자체 서명 인증서 생성 중..."
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout ssl/key.pem -out ssl/cert.pem \
                -subj "/C=KR/ST=Seoul/L=Seoul/O=TechAnalysis/CN=localhost"
            log_success "자체 서명 인증서 생성 완료"
        fi
    fi
    
    # 환경 변수 설정
    if [ ! -f ".env.production" ]; then
        log_info "프로덕션 환경 변수 파일 생성 중..."
        cat > .env.production << EOF
# 프로덕션 환경 변수
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# 보안 설정
SECRET_KEY=$(openssl rand -hex 32)
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# 데이터베이스 (확장용)
DB_HOST=db
DB_PORT=5432
DB_NAME=tech_analysis
DB_USER=postgres
DB_PASSWORD=$(openssl rand -hex 16)

# 외부 서비스
REDIS_URL=redis://tech-analysis-redis:6379/0
SENTRY_DSN=your_sentry_dsn_here

# 모니터링
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
EOF
        log_success "프로덕션 환경 변수 파일 생성 완료"
    fi
    
    # 프로덕션용 Docker Compose 시작
    log_info "프로덕션 서비스 시작 중..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    
    log_success "프로덕션 환경 설정 완료"
    log_info "HTTPS 접속: https://your-domain.com"
    log_info "HTTP는 자동으로 HTTPS로 리다이렉트됩니다."
}

# 테스트 실행
run_tests() {
    log_info "테스트 실행 중..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # 유닛 테스트
    if command -v pytest &> /dev/null; then
        log_info "유닛 테스트 실행 중..."
        pytest tests/unit/ -v
        log_success "유닛 테스트 완료"
    else
        log_warning "pytest가 설치되어 있지 않습니다. pip install pytest로 설치해주세요."
    fi
    
    # 코드 품질 검사
    if command -v flake8 &> /dev/null; then
        log_info "코드 품질 검사 실행 중..."
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        log_success "코드 품질 검사 완료"
    fi
    
    log_success "모든 테스트 완료"
}

# 도움말 출력
show_help() {
    cat << EOF
사용법: $0 [옵션]

옵션:
  local       로컬 개발 환경 설정
  docker      Docker 환경 설정
  production  프로덕션 환경 설정
  test        테스트 실행
  clean       임시 파일 정리
  help        이 도움말 출력

예시:
  $0 local      # 로컬 개발 환경 설정
  $0 docker     # Docker로 실행
  $0 production # 프로덕션 배포

추가 정보:
  - README.md 파일을 참조하세요
  - 문제 발생 시 GitHub Issues에 신고해주세요
  - 문서: https://github.com/your-username/global-tech-analysis
EOF
}

# 정리 함수
cleanup() {
    log_info "임시 파일 정리 중..."
    
    # Python 캐시 정리
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    
    # 캐시 디렉토리 정리
    rm -rf cache/*
    
    # 로그 파일 정리 (7일 이상)
    find logs/ -name "*.log*" -mtime +7 -delete 2>/dev/null || true
    
    # Docker 정리 (사용하지 않는 이미지, 컨테이너, 네트워크)
    if command -v docker &> /dev/null; then
        docker system prune -f
    fi
    
    log_success "정리 완료"
}

# 메인 함수
main() {
    print_banner
    
    case "${1:-help}" in
        "local")
            check_requirements
            create_directories
            setup_local
            log_success "로컬 환경 설정이 완료되었습니다!"
            log_info "실행 명령어: source venv/bin/activate && streamlit run app.py"
            ;;
        "docker")
            check_docker_requirements
            create_directories
            setup_docker
            log_success "Docker 환경 설정이 완료되었습니다!"
            log_info "접속 URL: http://localhost:8501"
            ;;
        "production")
            check_docker_requirements
            create_directories
            setup_production
            log_success "프로덕션 환경 설정이 완료되었습니다!"
            log_info "서비스가 시작되었습니다. 도메인을 통해 접속하세요."
            ;;
        "test")
            run_tests
            ;;
        "clean")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# 스크립트 실행
main "$@"
