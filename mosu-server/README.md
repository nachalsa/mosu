# MOSU Server

MOSU 실시간 수화 인식 서버입니다.

## 🎯 주요 기능

- 🎥 **실시간 웹캠 입력**: 브라우저를 통한 웹캠 스트리밍
- 🤖 **포즈 추정**: 133개 키포인트 실시간 감지
- 🤲 **수화 인식**: Transformer 기반 딥러닝 모델
- 🌐 **WebSocket 통신**: 실시간 양방향 통신  
- 📱 **웹 UI**: 반응형 웹 인터페이스
- 📊 **실시간 통계**: FPS, 신뢰도, 추론 시간 표시

## 🚀 빠른 시작

### 1. 간단한 관리 도구 사용 (추천)

```bash
cd /home/lts/gitwork/mosu/mosu-server
./manage.sh
```

메뉴에서 원하는 서버를 선택하여 실행하세요:
- **옵션 1**: 간단한 테스트 서버 (더미 모델, 포트 8001)
- **옵션 2**: 실제 모델 서버 (Transformer 모델, 포트 8002)

### 2. 직접 실행

#### 개발/테스트용 간단한 서버
```bash
./start_server.sh
# 또는
python simple_server.py --port 8001
```

#### 실제 수화 모델 서버  
```bash
./start_real_server.sh
# 또는
python real_server.py --model ../mosumodel/best_model_stage_1.pt --port 8002
```

## 📋 시스템 요구사항

### 필수 패키지
```bash
pip install fastapi uvicorn websockets pillow numpy opencv-python torch
```

### 선택사항 (고급 기능)
```bash
pip install mmpose mmcv mmdet mmengine  # RTMW 포즈 추정용
```

## 🖥️ 서버 종류

### 1. 간단한 테스트 서버 (포트: 8001)
- **용도**: 개발 및 테스트
- **특징**: 더미 포즈 추정기, 더미 수화 인식기 사용
- **장점**: 의존성 최소, 빠른 시작
- **접속**: http://localhost:8001

### 2. 실제 모델 서버 (포트: 8002)  
- **용도**: 실제 수화 인식
- **특징**: 실제 Transformer 모델 사용
- **장점**: 정확한 수화 인식
- **접속**: http://localhost:8002

## 📖 사용법

1. 서버 실행 후 브라우저에서 접속
2. **"🚀 시스템 시작"** 버튼 클릭
3. 웹캠 권한 허용
4. 손으로 수화 동작 수행
5. 실시간으로 포즈 추정 및 수화 인식 결과 확인

## 🔧 고급 설정

### 명령행 옵션

```bash
python real_server.py --help
```

- `--model`: 수화 인식 모델 경로
- `--device`: 추론 디바이스 (auto/cuda/xpu/cpu)  
- `--host`: 서버 호스트 (기본값: 0.0.0.0)
- `--port`: 서버 포트 (기본값: 8002)

### 환경별 실행

```bash
# CPU만 사용
python real_server.py --device cpu

# CUDA GPU 사용  
python real_server.py --device cuda

# Intel XPU 사용
python real_server.py --device xpu
```

## 🔗 API 엔드포인트

### REST API
- `GET /`: 메인 웹 인터페이스
- `GET /health`: 서버 상태 및 모델 정보
- `GET /stats`: 상세 통계 (추론 시간, 감지된 단어 등)

### WebSocket  
- `WebSocket /ws`: 실시간 이미지 전송 및 결과 수신

### WebSocket 프로토콜

**클라이언트 → 서버:**
```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "timestamp": 1692547200000
}
```

**서버 → 클라이언트:**
```json
{
  "type": "result", 
  "timestamp": 1692547200.123,
  "pose": {
    "keypoints": [[x1,y1], [x2,y2], ...],
    "scores": [0.95, 0.87, ...]
  },
  "sign": {
    "word": "안녕하세요",
    "confidence": 0.89
  },
  "stats": {
    "avg_inference_time": 0.045
  }
}
```

## 🛠️ 문제 해결

### 일반적인 문제

**Q: 웹캠에 접근할 수 없습니다**
- A: 브라우저에서 웹캠 권한을 허용했는지 확인하세요
- A: HTTPS가 필요한 경우 로컬호스트에서 테스트하세요

**Q: 모델 로딩 실패**  
- A: 모델 파일 경로가 올바른지 확인: `../mosumodel/best_model_stage_1.pt`
- A: PyTorch가 설치되었는지 확인하세요

**Q: 포즈가 제대로 감지되지 않음**
- A: 조명이 충분한지 확인하세요
- A: 카메라가 전신을 담고 있는지 확인하세요

**Q: 수화 인식 정확도가 낮음**
- A: 모델이 학습된 수화 동작과 유사하게 수행하세요  
- A: 손동작이 카메라에 잘 보이도록 위치를 조정하세요

### 로그 확인

서버 실행 시 콘솔에서 다음과 같은 로그를 확인할 수 있습니다:

```
🚀 MOSU 서버 시작 (실제 모델)
   - 주소: http://0.0.0.0:8002
   - 모델: 실제
   - Ctrl+C로 종료

🔗 새 연결: 총 1개
🎯 수화 인식: 안녕하세요 (신뢰도: 0.874)
```

## 📁 파일 구조

```
mosu-server/
├── main.py              # 완전한 MOSU 서버 (RTMW + 수화모델)
├── simple_server.py     # 간단한 테스트 서버 (더미 모델)
├── real_server.py       # 실제 수화 모델 서버
├── test_server.py       # 기본 동작 테스트용
├── manage.sh           # 서버 관리 도구 
├── start_server.sh     # 간단한 서버 시작 스크립트
├── start_real_server.sh # 실제 서버 시작 스크립트
├── requirements.txt     # 필수 패키지 목록
└── README.md           # 이 파일
```

## 🤝 기여하기

버그 리포트, 기능 요청, 또는 개선 사항이 있으시면 이슈를 생성해 주세요.

## 📄 라이센스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.
