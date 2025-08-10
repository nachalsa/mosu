# MOSU Server

MOSU 실시간 수화 인식 서버입니다.

## 기능

- 실시간 웹캠 입력 수신
- RTMW 기반 포즈 추정
- Transformer 기반 수화 인식
- WebSocket을 통한 실시간 결과 전송
- 웹 기반 사용자 인터페이스

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. MMPose 설치 (선택사항)

실제 포즈 추정을 위해서는 MMPose가 필요합니다:

```bash
pip install mmpose mmcv mmdet mmengine
```

### 3. RTMW 모델 다운로드 (선택사항)

포즈 추정 모델을 다운로드하여 `models/` 폴더에 배치하거나, 기본 경로에 설정하세요.

## 실행

### 방법 1: 완전한 서버 (RTMW + 수화 모델)

```bash
python main.py --model ../mosumodel/best_model_stage_1.pt
```

### 방법 2: 개발용 간단한 서버 (더미 모델)

```bash
python simple_server.py
```

### 방법 3: 실행 스크립트 사용

```bash
python run_server.py
```

## 옵션

- `--model`: 수화 인식 모델 경로 (기본값: ../mosumodel/best_model_stage_1.pt)
- `--device`: 추론 디바이스 (auto/cuda/xpu/cpu, 기본값: auto)
- `--host`: 서버 호스트 (기본값: 0.0.0.0)
- `--port`: 서버 포트 (기본값: 8000)

## 사용법

1. 서버 실행 후 브라우저에서 `http://localhost:8000` 접속
2. "카메라 시작" 버튼 클릭
3. 웹캠이 활성화되면 실시간 수화 인식 시작
4. 오른쪽 화면에서 포즈 추정 결과 확인
5. 하단에서 인식된 단어 확인

## API 엔드포인트

- `GET /`: 메인 웹 인터페이스
- `GET /health`: 서버 상태 확인
- `GET /stats`: 통계 정보
- `WebSocket /ws`: 실시간 통신

## WebSocket 프로토콜

### 클라이언트 → 서버

```json
{
  "type": "frame",
  "image": "data:image/jpeg;base64,..."
}
```

### 서버 → 클라이언트

```json
{
  "type": "result",
  "timestamp": 1234567890.123,
  "pose": {
    "keypoints": [[x1, y1], [x2, y2], ...],
    "scores": [score1, score2, ...]
  },
  "sign": {
    "word": "안녕하세요",
    "confidence": 0.85
  }
}
```

## 주의사항

- RTMW 모델 파일이 없으면 더미 포즈 추정기가 사용됩니다
- 수화 인식 모델 파일이 필요합니다 (`best_model_stage_1.pt`)
- 웹캠 접근 권한이 필요합니다
- HTTPS 환경에서는 보안 정책에 따라 웹캠 접근이 제한될 수 있습니다

## 개발 모드

개발 및 테스트를 위해 `simple_server.py`를 사용하면 모든 의존성 없이 서버를 실행할 수 있습니다:

```bash
python simple_server.py --port 8001
```

이 모드에서는:
- 더미 포즈 추정기 사용
- 더미 수화 인식기 사용 (랜덤하게 단어 생성)
- 최소한의 의존성만 필요
