#!/usr/bin/env python3
"""
MOSU 실시간 수화 인식 서버
- 웹캠 입력 수신
- RTMW 포즈 추정
- 수화 인식 추론
- 실시간 결과 반환
"""

import asyncio
import cv2
import numpy as np
import torch
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import base64
import io
from PIL import Image

# FastAPI & WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
import uvicorn

# MMPose 관련
from mmpose.apis import init_model, inference_topdown

# 프로젝트 모듈
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sign_language_model import SequenceToSequenceSignModel, RealtimeDecoder

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RTMWPoseEstimator:
    """RTMW 포즈 추정기"""
    
    def __init__(self, 
                 rtmw_config: str = None,
                 rtmw_checkpoint: str = None,
                 device: str = "auto"):
        
        # 기본 경로 설정 (pose-server에서 사용하는 경로와 동일)
        if rtmw_config is None:
            rtmw_config = "configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py"
        if rtmw_checkpoint is None:
            rtmw_checkpoint = "models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-f840f204_20231122.pth"
        
        self.rtmw_config = rtmw_config
        self.rtmw_checkpoint = rtmw_checkpoint
        self.device = self._determine_device(device)
        
        # PyTorch 보안 설정
        self.original_load = torch.load
        torch.load = lambda *args, **kwargs: self.original_load(
            *args, **kwargs, weights_only=False
        ) if 'weights_only' not in kwargs else self.original_load(*args, **kwargs)
        
        self._initialize_model()
    
    def _determine_device(self, device: str) -> str:
        """디바이스 자동 결정"""
        if device == "auto":
            try:
                if torch.xpu.is_available():
                    return "xpu"
            except:
                pass
            
            if torch.cuda.is_available():
                return "cuda"
            
            return "cpu"
        return device
    
    def _initialize_model(self):
        """RTMW 모델 초기화"""
        logger.info(f"🔧 RTMW 포즈 모델 로딩 중... (디바이스: {self.device})")
        start_time = time.time()
        
        try:
            # 설정 파일이 없으면 더미 포즈 추정기 사용
            if not Path(self.rtmw_config).exists() or not Path(self.rtmw_checkpoint).exists():
                logger.warning("⚠️ RTMW 모델 파일을 찾을 수 없어 더미 포즈 추정기를 사용합니다")
                self.use_dummy = True
                self.pose_model = None
            else:
                self.pose_model = init_model(
                    config=self.rtmw_config,
                    checkpoint=self.rtmw_checkpoint,
                    device=self.device
                )
                self.use_dummy = False
            
            init_time = time.time() - start_time
            logger.info(f"✅ 포즈 모델 로딩 완료: {init_time:.2f}초")
            
        except Exception as e:
            logger.warning(f"⚠️ RTMW 모델 로딩 실패, 더미 포즈 추정기 사용: {e}")
            self.use_dummy = True
            self.pose_model = None
    
    def estimate_pose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """이미지에서 포즈 추정"""
        if self.use_dummy:
            return self._dummy_pose_estimation(image)
        
        try:
            h, w = image.shape[:2]
            full_bbox = [0, 0, w, h]
            
            results = inference_topdown(
                model=self.pose_model,
                img=image,
                bboxes=[full_bbox],
                bbox_format='xyxy'
            )
            
            if results and len(results) > 0:
                keypoints = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                
                return keypoints, scores
            else:
                return self._dummy_pose_estimation(image)
                
        except Exception as e:
            logger.error(f"❌ 포즈 추정 실패: {e}")
            return self._dummy_pose_estimation(image)
    
    def _dummy_pose_estimation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """더미 포즈 추정 (개발용)"""
        h, w = image.shape[:2]
        
        # 더미 키포인트 생성 (133개)
        keypoints = np.random.rand(133, 2) * np.array([w, h])
        scores = np.random.rand(133) * 0.8 + 0.2  # 0.2-1.0 범위
        
        return keypoints, scores

class SignLanguageInferencer:
    """수화 인식 추론기"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 confidence_threshold: float = 0.7):
        
        self.device = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        self._load_model(model_path)
        self._initialize_components()
        
        logger.info(f"✅ 수화 인식 모델 로딩 완료 (디바이스: {self.device})")
    
    def _load_model(self, model_path: str):
        """수화 인식 모델 로드"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Vocabulary 추출
            if 'vocab_words' in checkpoint:
                self.words = checkpoint['vocab_words']
                self.word_to_id = {word: i for i, word in enumerate(self.words)}
            else:
                # 기본 vocabulary
                self.words = [f"단어_{i:03d}" for i in range(442)]
                self.word_to_id = {word: i for i, word in enumerate(self.words)}
            
            vocab_size = len(self.words)
            
            # 모델 생성
            self.model = SequenceToSequenceSignModel(
                vocab_size=vocab_size,
                embed_dim=256,
                num_encoder_layers=6,
                num_decoder_layers=4,
                num_heads=8
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"📚 수화 모델 로드 완료: vocabulary {vocab_size}개 단어")
            
        except Exception as e:
            logger.error(f"❌ 수화 모델 로드 실패: {e}")
            raise
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        self.window_size = 60  # 2초 (30fps 기준)
        self.pose_buffer = deque(maxlen=self.window_size)
        
        self.decoder = RealtimeDecoder(
            vocab_size=len(self.words),
            confidence_threshold=self.confidence_threshold
        )
        
        # 통계
        self.inference_times = deque(maxlen=100)
        self.detected_words = []
    
    @torch.no_grad()
    def process_pose(self, keypoints: np.ndarray, scores: np.ndarray) -> Optional[str]:
        """포즈 데이터 처리 및 수화 인식"""
        # 포즈 특징 생성 [133, 3] (x, y, score)
        pose_features = np.zeros((133, 3), dtype=np.float32)
        pose_features[:, :2] = keypoints
        pose_features[:, 2] = scores
        
        # 버퍼에 추가
        self.pose_buffer.append(pose_features)
        
        # 최소 프레임 수 확인
        if len(self.pose_buffer) < 30:  # 1초 최소
            return None
        
        start_time = time.time()
        
        # 윈도우 데이터 준비
        window_data = np.array(list(self.pose_buffer))  # [frames, 133, 3]
        
        # 패딩 또는 자르기
        if len(window_data) < self.window_size:
            padding = np.zeros((self.window_size - len(window_data), 133, 3), dtype=np.float32)
            window_data = np.concatenate([padding, window_data], axis=0)
        
        # 텐서 변환 및 추론
        input_tensor = torch.from_numpy(window_data).unsqueeze(0).to(self.device)
        
        try:
            outputs = self.model(input_tensor)
            
            # 현재 프레임의 출력
            word_logits = outputs['word_logits'][0, -1]
            boundary_logits = outputs['boundary_logits'][0, -1]
            confidence_score = outputs['confidence_scores'][0, -1]
            
            # 실시간 디코더 처리
            result = self.decoder.process_frame_output(word_logits, boundary_logits, confidence_score)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            if result is not None:
                word = self.words[result] if result < len(self.words) else f"단어_{result}"
                self.detected_words.append(word)
                logger.info(f"🎯 수화 인식: {word}")
                return word
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 수화 추론 실패: {e}")
            return None

class MosuServer:
    """MOSU 통합 서버"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 host: str = "0.0.0.0",
                 port: int = 8000):
        
        self.host = host
        self.port = port
        
        # 컴포넌트 초기화
        self.pose_estimator = RTMWPoseEstimator(device=device)
        self.sign_inferencer = SignLanguageInferencer(model_path, device)
        
        # FastAPI 앱 설정
        self.app = FastAPI(title="MOSU Sign Language Recognition Server")
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # WebSocket 연결 관리
        self.connections: List[WebSocket] = []
        
        self.setup_routes()
        
        logger.info(f"🚀 MOSU 서버 초기화 완료")
        logger.info(f"   - 서버 주소: {host}:{port}")
        logger.info(f"   - Vocabulary: {len(self.sign_inferencer.words)}개 단어")
    
    def setup_routes(self):
        """라우트 설정"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """메인 페이지"""
            return self.get_index_html()
        
        @self.app.get("/health")
        async def health_check():
            """헬스 체크"""
            return {
                "status": "healthy",
                "pose_device": self.pose_estimator.device,
                "sign_device": str(self.sign_inferencer.device),
                "vocab_size": len(self.sign_inferencer.words),
                "connections": len(self.connections)
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """통계 정보"""
            pose_times = getattr(self.pose_estimator, 'inference_times', [])
            sign_times = list(self.sign_inferencer.inference_times)
            
            return {
                "connections": len(self.connections),
                "detected_words": self.sign_inferencer.detected_words[-10:],  # 최근 10개
                "pose_estimation": {
                    "count": len(pose_times),
                    "avg_time": np.mean(pose_times) if pose_times else 0
                },
                "sign_recognition": {
                    "count": len(sign_times),
                    "avg_time": np.mean(sign_times) if sign_times else 0
                }
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 엔드포인트"""
            await websocket.accept()
            self.connections.append(websocket)
            logger.info(f"🔗 새 연결: 총 {len(self.connections)}개")
            
            try:
                while True:
                    # 클라이언트로부터 이미지 데이터 수신
                    data = await websocket.receive_json()
                    
                    if data["type"] == "frame":
                        # Base64 이미지 디코딩
                        image_data = base64.b64decode(data["image"].split(",")[1])
                        image = Image.open(io.BytesIO(image_data))
                        image_np = np.array(image)
                        
                        # RGB to BGR (OpenCV)
                        if len(image_np.shape) == 3:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        # 포즈 추정
                        keypoints, scores = self.pose_estimator.estimate_pose(image_np)
                        
                        # 수화 인식
                        detected_word = self.sign_inferencer.process_pose(keypoints, scores)
                        
                        # 결과 전송
                        response = {
                            "type": "result",
                            "timestamp": time.time(),
                            "pose": {
                                "keypoints": keypoints.tolist(),
                                "scores": scores.tolist()
                            },
                            "sign": {
                                "word": detected_word,
                                "confidence": float(scores.mean()) if scores is not None else 0.0
                            }
                        }
                        
                        await websocket.send_json(response)
                        
            except WebSocketDisconnect:
                self.connections.remove(websocket)
                logger.info(f"🔌 연결 종료: 총 {len(self.connections)}개")
            except Exception as e:
                logger.error(f"❌ WebSocket 오류: {e}")
                if websocket in self.connections:
                    self.connections.remove(websocket)
    
    def get_index_html(self) -> str:
        """메인 페이지 HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MOSU 실시간 수화 인식</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .video-container { display: flex; gap: 20px; margin-bottom: 20px; }
                .video-box { flex: 1; }
                video, canvas { width: 100%; max-width: 640px; height: 480px; border: 2px solid #ddd; border-radius: 10px; }
                .controls { text-align: center; margin: 20px 0; }
                button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; border: none; border-radius: 5px; }
                .start-btn { background-color: #4CAF50; color: white; }
                .stop-btn { background-color: #f44336; color: white; }
                .results { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .word-display { font-size: 24px; font-weight: bold; color: #2196F3; margin: 10px 0; }
                .stats { display: flex; gap: 20px; margin-top: 20px; }
                .stat-box { flex: 1; background: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }
                #detected-words { min-height: 100px; background: #f5f5f5; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤲 MOSU 실시간 수화 인식</h1>
                    <p>웹캠을 사용한 실시간 수화 인식 시스템</p>
                </div>
                
                <div class="video-container">
                    <div class="video-box">
                        <h3>📹 웹캠 입력</h3>
                        <video id="video" autoplay muted></video>
                    </div>
                    <div class="video-box">
                        <h3>🎯 포즈 추정</h3>
                        <canvas id="canvas"></canvas>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="start-btn" onclick="startCamera()">📹 카메라 시작</button>
                    <button class="stop-btn" onclick="stopCamera()">⏹️ 정지</button>
                    <button onclick="clearResults()">🗑️ 결과 초기화</button>
                </div>
                
                <div class="results">
                    <h3>🎯 인식 결과</h3>
                    <div class="word-display" id="current-word">대기 중...</div>
                    
                    <div class="stats">
                        <div class="stat-box">
                            <h4>📊 연결 상태</h4>
                            <div id="connection-status">연결 안됨</div>
                        </div>
                        <div class="stat-box">
                            <h4>⚡ 처리 속도</h4>
                            <div id="fps">0 FPS</div>
                        </div>
                        <div class="stat-box">
                            <h4>🎯 신뢰도</h4>
                            <div id="confidence">0%</div>
                        </div>
                    </div>
                    
                    <h4>📝 감지된 단어들</h4>
                    <div id="detected-words"></div>
                </div>
            </div>
            
            <script>
                let video = document.getElementById('video');
                let canvas = document.getElementById('canvas');
                let ctx = canvas.getContext('2d');
                let ws = null;
                let stream = null;
                let sending = false;
                let frameCount = 0;
                let lastFpsUpdate = Date.now();
                let detectedWords = [];
                
                // WebSocket 연결
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        document.getElementById('connection-status').textContent = '✅ 연결됨';
                        console.log('WebSocket 연결 성공');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'result') {
                            // 포즈 시각화
                            drawPose(data.pose);
                            
                            // 수화 결과 표시
                            if (data.sign.word) {
                                document.getElementById('current-word').textContent = data.sign.word;
                                addDetectedWord(data.sign.word);
                            }
                            
                            // 신뢰도 업데이트
                            const confidence = Math.round(data.sign.confidence * 100);
                            document.getElementById('confidence').textContent = confidence + '%';
                            
                            // FPS 계산
                            frameCount++;
                            const now = Date.now();
                            if (now - lastFpsUpdate > 1000) {
                                const fps = Math.round(frameCount * 1000 / (now - lastFpsUpdate));
                                document.getElementById('fps').textContent = fps + ' FPS';
                                frameCount = 0;
                                lastFpsUpdate = now;
                            }
                        }
                    };
                    
                    ws.onclose = function() {
                        document.getElementById('connection-status').textContent = '❌ 연결 끊김';
                        console.log('WebSocket 연결 종료');
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket 오류:', error);
                        document.getElementById('connection-status').textContent = '❌ 오류';
                    };
                }
                
                // 카메라 시작
                async function startCamera() {
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({
                            video: { width: 640, height: 480 }
                        });
                        video.srcObject = stream;
                        
                        canvas.width = 640;
                        canvas.height = 480;
                        
                        if (!ws || ws.readyState === WebSocket.CLOSED) {
                            connectWebSocket();
                        }
                        
                        // 프레임 전송 시작
                        setTimeout(sendFrame, 100); // 10 FPS로 제한
                        
                    } catch (error) {
                        console.error('카메라 접근 오류:', error);
                        alert('카메라에 접근할 수 없습니다.');
                    }
                }
                
                // 카메라 정지
                function stopCamera() {
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                        stream = null;
                    }
                    sending = false;
                    
                    if (ws) {
                        ws.close();
                    }
                }
                
                // 프레임 전송
                function sendFrame() {
                    if (!stream || !ws || ws.readyState !== WebSocket.OPEN || sending) {
                        if (stream) setTimeout(sendFrame, 100);
                        return;
                    }
                    
                    sending = true;
                    
                    // 임시 캔버스에 비디오 프레임 그리기
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = 640;
                    tempCanvas.height = 480;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(video, 0, 0, 640, 480);
                    
                    // Base64로 인코딩하여 전송
                    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7);
                    
                    ws.send(JSON.stringify({
                        type: 'frame',
                        image: dataUrl
                    }));
                    
                    sending = false;
                    setTimeout(sendFrame, 100); // 10 FPS
                }
                
                // 포즈 시각화
                function drawPose(pose) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    const keypoints = pose.keypoints;
                    const scores = pose.scores;
                    
                    // 키포인트 그리기
                    ctx.fillStyle = 'red';
                    for (let i = 0; i < keypoints.length; i++) {
                        if (scores[i] > 0.5) {
                            const x = keypoints[i][0];
                            const y = keypoints[i][1];
                            ctx.beginPath();
                            ctx.arc(x, y, 3, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    }
                }
                
                // 감지된 단어 추가
                function addDetectedWord(word) {
                    if (!detectedWords.includes(word)) {
                        detectedWords.push(word);
                        updateDetectedWordsDisplay();
                    }
                }
                
                // 감지된 단어 표시 업데이트
                function updateDetectedWordsDisplay() {
                    const container = document.getElementById('detected-words');
                    container.innerHTML = detectedWords.map(word => 
                        `<span style="display: inline-block; margin: 5px; padding: 5px 10px; background: #2196F3; color: white; border-radius: 15px;">${word}</span>`
                    ).join('');
                }
                
                // 결과 초기화
                function clearResults() {
                    detectedWords = [];
                    document.getElementById('current-word').textContent = '대기 중...';
                    document.getElementById('detected-words').innerHTML = '';
                    document.getElementById('confidence').textContent = '0%';
                }
                
                // 페이지 로드 시 WebSocket 연결
                window.onload = function() {
                    connectWebSocket();
                };
                
                // 페이지 종료 시 정리
                window.onbeforeunload = function() {
                    stopCamera();
                };
            </script>
        </body>
        </html>
        """
    
    def run(self):
        """서버 실행"""
        logger.info(f"\n🚀 MOSU 서버 시작")
        logger.info(f"   - 주소: http://{self.host}:{self.port}")
        logger.info(f"   - 헬스체크: http://{self.host}:{self.port}/health")
        logger.info(f"   - 통계: http://{self.host}:{self.port}/stats")
        logger.info(f"   - Ctrl+C로 종료")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("\n⏹️ MOSU 서버 종료")
        except Exception as e:
            logger.error(f"❌ 서버 실행 실패: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MOSU Real-time Sign Language Recognition Server")
    parser.add_argument("--model", type=str, 
                       default="../mosumodel/best_model_stage_1.pt",
                       help="수화 인식 모델 경로")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "xpu"],
                       help="추론 디바이스")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000,
                       help="서버 포트")
    
    args = parser.parse_args()
    
    # 모델 파일 존재 확인
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    try:
        server = MosuServer(
            model_path=str(model_path),
            device=args.device,
            host=args.host,
            port=args.port
        )
        
        server.run()
        
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")

if __name__ == "__main__":
    main()
