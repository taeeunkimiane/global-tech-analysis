// tests/performance/load-test.js
// K6를 사용한 글로벌 기술 이슈 분석 시스템 성능 테스트

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// 커스텀 메트릭
const errorRate = new Rate('errors');
const responseTimeTrend = new Trend('response_time');
const requestCounter = new Counter('requests_total');

// 테스트 환경 설정
const BASE_URL = __ENV.TEST_HOST || 'http://localhost:8501';

// 테스트 옵션
export const options = {
  stages: [
    // 워밍업: 5분 동안 0명에서 10명으로 증가
    { duration: '5m', target: 10 },
    // 안정 상태: 10분 동안 10명 유지
    { duration: '10m', target: 10 },
    // 로드 증가: 5분 동안 10명에서 50명으로 증가
    { duration: '5m', target: 50 },
    // 피크 로드: 10분 동안 50명 유지
    { duration: '10m', target: 50 },
    // 스파이크 테스트: 2분 동안 100명으로 급증
    { duration: '2m', target: 100 },
    // 복구: 5분 동안 100명에서 10명으로 감소
    { duration: '5m', target: 10 },
    // 쿨다운: 5분 동안 10명에서 0명으로 감소
    { duration: '5m', target: 0 },
  ],
  thresholds: {
    // 에러율이 5% 미만이어야 함
    errors: ['rate<0.05'],
    // 95%의 응답시간이 5초 미만이어야 함
    'http_req_duration': ['p(95)<5000'],
    // 평균 응답시간이 2초 미만이어야 함
    'http_req_duration{type:main_page}': ['avg<2000'],
    // 헬스체크는 항상 200ms 미만이어야 함
    'http_req_duration{type:health}': ['p(99)<200'],
  },
};

// 테스트 시나리오 가중치
const scenarios = [
  { name: 'main_page', weight: 40 },
  { name: 'filter_countries', weight: 20 },
  { name: 'filter_categories', weight: 15 },
  { name: 'view_insights', weight: 15 },
  { name: 'health_check', weight: 10 },
];

// 국가 및 카테고리 목록 (실제 애플리케이션과 동일)
const countries = ['미국', '중국', '일본', '독일', '영국', '프랑스', '한국'];
const categories = ['AI/ML', '하드웨어 혁신', '보안/해킹', '법률/규제', '자율시스템'];

// 랜덤 선택 유틸리티
function randomChoice(array) {
  return array[Math.floor(Math.random() * array.length)];
}

function randomChoices(array, count) {
  const shuffled = array.sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
}

// 시나리오 선택 함수
function selectScenario() {
  const random = Math.random() * 100;
  let cumulative = 0;
  
  for (const scenario of scenarios) {
    cumulative += scenario.weight;
    if (random <= cumulative) {
      return scenario.name;
    }
  }
  return scenarios[0].name;
}

// 메인 페이지 테스트
function testMainPage() {
  const response = http.get(`${BASE_URL}/`, {
    tags: { type: 'main_page' },
    timeout: '30s',
  });
  
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'contains header': (r) => r.body.includes('글로벌 기술 이슈 분석 시스템'),
    'loads within 5s': (r) => r.timings.duration < 5000,
    'content length > 1000': (r) => r.body.length > 1000,
  });
  
  errorRate.add(!success);
  responseTimeTrend.add(response.timings.duration);
  requestCounter.add(1);
  
  return response;
}

// 국가 필터 테스트
function testCountryFilter() {
  const selectedCountries = randomChoices(countries, Math.floor(Math.random() * 3) + 1);
  
  const response = http.post(`${BASE_URL}/`, {
    // Streamlit 앱의 실제 필터 동작을 시뮬레이션
    // 실제로는 WebSocket 통신이지만 HTTP POST로 근사
  }, {
    tags: { type: 'filter_countries' },
    timeout: '15s',
  });
  
  const success = check(response, {
    'status is 200 or 204': (r) => [200, 204].includes(r.status),
    'response time < 3s': (r) => r.timings.duration < 3000,
  });
  
  errorRate.add(!success);
  requestCounter.add(1);
  
  return response;
}

// 카테고리 필터 테스트
function testCategoryFilter() {
  const selectedCategories = randomChoices(categories, Math.floor(Math.random() * 3) + 1);
  
  const response = http.post(`${BASE_URL}/`, {
    // 카테고리 필터 요청 시뮬레이션
  }, {
    tags: { type: 'filter_categories' },
    timeout: '15s',
  });
  
  const success = check(response, {
    'status is 200 or 204': (r) => [200, 204].includes(r.status),
    'response time < 3s': (r) => r.timings.duration < 3000,
  });
  
  errorRate.add(!success);
  requestCounter.add(1);
  
  return response;
}

// 인사이트 보기 테스트
function testViewInsights() {
  const response = http.get(`${BASE_URL}/`, {
    tags: { type: 'insights' },
    timeout: '20s',
  });
  
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'contains insights': (r) => r.body.includes('인사이트') || r.body.includes('분석'),
    'loads within 10s': (r) => r.timings.duration < 10000,
  });
  
  errorRate.add(!success);
  requestCounter.add(1);
  
  return response;
}

// 헬스체크 테스트
function testHealthCheck() {
  const response = http.get(`${BASE_URL}/_stcore/health`, {
    tags: { type: 'health' },
    timeout: '5s',
  });
  
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
    'body is ok': (r) => r.body.length > 0,
  });
  
  errorRate.add(!success);
  requestCounter.add(1);
  
  return response;
}

// WebSocket 연결 테스트 (Streamlit의 실시간 업데이트)
function testWebSocketConnection() {
  const response = http.get(`${BASE_URL}/_stcore/stream`, {
    tags: { type: 'websocket' },
    headers: {
      'Upgrade': 'websocket',
      'Connection': 'Upgrade',
    },
    timeout: '10s',
  });
  
  const success = check(response, {
    'websocket connection possible': (r) => [101, 200].includes(r.status),
  });
  
  errorRate.add(!success);
  requestCounter.add(1);
  
  return response;
}

// 메인 테스트 함수
export default function() {
  const scenario = selectScenario();
  
  try {
    switch(scenario) {
      case 'main_page':
        testMainPage();
        break;
      case 'filter_countries':
        testCountryFilter();
        break;
      case 'filter_categories':
        testCategoryFilter();
        break;
      case 'view_insights':
        testViewInsights();
        break;
      case 'health_check':
        testHealthCheck();
        break;
      default:
        testMainPage();
    }
    
    // 사용자 행동 시뮬레이션 (페이지 간 대기시간)
    sleep(Math.random() * 3 + 1); // 1-4초 대기
    
  } catch (error) {
    console.error(`Test error in scenario ${scenario}:`, error);
    errorRate.add(1);
  }
}

// 테스트 시작 시 실행
export function setup() {
  console.log('🚀 성능 테스트 시작');
  console.log(`대상 URL: ${BASE_URL}`);
  
  // 기본 연결 확인
  const response = http.get(`${BASE_URL}/_stcore/health`);
  
  if (response.status !== 200) {
    throw new Error(`애플리케이션이 실행되고 있지 않습니다. 상태 코드: ${response.status}`);
  }
  
  console.log('✅ 애플리케이션 연결 확인 완료');
  return { timestamp: new Date().toISOString() };
}

// 테스트 종료 시 실행
export function teardown(data) {
  console.log('📊 성능 테스트 완료');
  console.log(`시작 시간: ${data.timestamp}`);
  console.log(`종료 시간: ${new Date().toISOString()}`);
}

// 커스텀 메트릭 핸들링
export function handleSummary(data) {
  return {
    'performance-report.html': generateHtmlReport(data),
    'performance-results.json': JSON.stringify(data, null, 2),
  };
}

// HTML 리포트 생성
function generateHtmlReport(data) {
  const { metrics } = data;
  
  return `
<!DOCTYPE html>
<html>
<head>
    <title>글로벌 기술 이슈 분석 시스템 - 성능 테스트 결과</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #667eea; color: white; padding: 20px; border-radius: 5px; }
        .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #667eea; background: #f5f5f5; }
        .success { border-left-color: #4caf50; }
        .warning { border-left-color: #ff9800; }
        .error { border-left-color: #f44336; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌐 글로벌 기술 이슈 분석 시스템</h1>
        <h2>성능 테스트 결과 리포트</h2>
        <p>생성 시간: ${new Date().toLocaleString()}</p>
    </div>
    
    <h3>📊 주요 메트릭</h3>
    
    <div class="metric ${metrics.http_req_duration?.med < 2000 ? 'success' : 'warning'}">
        <strong>응답 시간</strong><br>
        평균: ${metrics.http_req_duration?.avg?.toFixed(2)}ms<br>
        중간값: ${metrics.http_req_duration?.med?.toFixed(2)}ms<br>
        95퍼센타일: ${metrics.http_req_duration?.['p(95)']?.toFixed(2)}ms
    </div>
    
    <div class="metric ${metrics.errors?.rate < 0.05 ? 'success' : 'error'}">
        <strong>에러율</strong><br>
        ${(metrics.errors?.rate * 100)?.toFixed(2)}%
    </div>
    
    <div class="metric">
        <strong>요청 수</strong><br>
        총 요청: ${metrics.http_reqs?.count}<br>
        초당 요청: ${metrics.http_reqs?.rate?.toFixed(2)} RPS
    </div>
    
    <div class="metric">
        <strong>가상 사용자</strong><br>
        최대 동시 사용자: ${metrics.vus_max?.value}<br>
        평균 동시 사용자: ${metrics.vus?.value}
    </div>
    
    <h3>📈 시나리오별 성능</h3>
    <table>
        <tr>
            <th>시나리오</th>
            <th>평균 응답시간</th>
            <th>95% 응답시간</th>
            <th>요청 수</th>
        </tr>
        <tr>
            <td>메인 페이지</td>
            <td>${metrics['http_req_duration{type:main_page}']?.avg?.toFixed(2) || 'N/A'}ms</td>
            <td>${metrics['http_req_duration{type:main_page}']?.['p(95)']?.toFixed(2) || 'N/A'}ms</td>
            <td>${metrics['http_reqs{type:main_page}']?.count || 'N/A'}</td>
        </tr>
        <tr>
            <td>헬스체크</td>
            <td>${metrics['http_req_duration{type:health}']?.avg?.toFixed(2) || 'N/A'}ms</td>
            <td>${metrics['http_req_duration{type:health}']?.['p(95)']?.toFixed(2) || 'N/A'}ms</td>
            <td>${metrics['http_reqs{type:health}']?.count || 'N/A'}</td>
        </tr>
    </table>
    
    <h3>🎯 성능 목표 달성도</h3>
    <div class="metric ${metrics.errors?.rate < 0.05 ? 'success' : 'error'}">
        에러율 < 5%: ${metrics.errors?.rate < 0.05 ? '✅ 달성' : '❌ 미달성'}
    </div>
    <div class="metric ${metrics.http_req_duration?.['p(95)'] < 5000 ? 'success' : 'error'}">
        95% 응답시간 < 5초: ${metrics.http_req_duration?.['p(95)'] < 5000 ? '✅ 달성' : '❌ 미달성'}
    </div>
    
    <h3>💡 권장사항</h3>
    <ul>
        ${metrics.http_req_duration?.avg > 2000 ? '<li>응답 시간이 높습니다. 캐싱 개선을 고려하세요.</li>' : ''}
        ${metrics.errors?.rate > 0.01 ? '<li>에러율이 높습니다. 로그를 확인하여 원인을 파악하세요.</li>' : ''}
        ${metrics.http_reqs?.rate < 10 ? '<li>처리량이 낮습니다. 서버 리소스를 확인하세요.</li>' : ''}
        <li>지속적인 모니터링을 통해 성능 저하를 조기에 감지하세요.</li>
        <li>정기적인 성능 테스트를 통해 회귀를 방지하세요.</li>
    </ul>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
        <p>이 리포트는 K6를 사용하여 자동 생성되었습니다.</p>
    </footer>
</body>
</html>
  `;
}
