#!/usr/bin/env python3
"""
MOSU 서버 - 실제 수화 모델 사용 버전
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# 프로젝트 모듈
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sign_language_model import SequenceToSequenceSignModel, RealtimeDecoder
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"⚠️ 수화 모델을 로드할 수 없습니다: {e}")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class RealSignLanguageInferencer:
    """실제 수화 인식 추론기"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = self._determine_device(device)
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        self._load_model()
        self._initialize_components()
        
        logger.info(f"✅ 실제 수화 모델 로딩 완료 (디바이스: {self.device})")
    
    def _determine_device(self, device: str) -> str:
        """디바이스 결정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            try:
                if torch.xpu.is_available():
                    return "xpu"
            except:
                pass
            return "cpu"
        return device
    
    def _load_model(self):
        """모델 로드"""
        try:
            # PyTorch 보안 설정
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(
                *args, **kwargs, weights_only=False
            ) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Vocabulary 추출
            if 'vocab_words' in checkpoint:
                self.words = checkpoint['vocab_words']
            elif 'words' in checkpoint:
                self.words = checkpoint['words']
            else:
                # 기본 한국어 수화 단어들
                self.words = [
                    "안녕하세요", "감사합니다", "죄송합니다", "사랑해요", "괜찮아요",
                    "도와주세요", "좋아요", "싫어요", "네", "아니요", "있어요", "없어요",
                    "먹다", "마시다", "가다", "오다", "보다", "듣다", "말하다", "자다",
                    "학교", "집", "병원", "회사", "친구", "가족", "엄마", "아빠",
                    "물", "밥", "책", "컴퓨터", "전화", "시간", "돈", "일",
                    "기쁘다", "슬프다", "화나다", "무섭다", "행복하다", "걱정하다"
                ]
                logger.warning("⚠️ 체크포인트에 vocabulary가 없어서 기본 단어를 사용합니다")
            
            self.word_to_id = {word: i for i, word in enumerate(self.words)}
            vocab_size = len(self.words)
            
            # 모델 아키텍처 생성
            model_config = checkpoint.get('model_config', {})
            self.model = SequenceToSequenceSignModel(
                vocab_size=vocab_size,
                embed_dim=model_config.get('embed_dim', 256),
                num_encoder_layers=model_config.get('num_encoder_layers', 6),
                num_decoder_layers=model_config.get('num_decoder_layers', 4),
                num_heads=model_config.get('num_heads', 8)
            )
            
            # 모델 가중치 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"📚 수화 모델 로드 완료: {vocab_size}개 단어")
            
        except Exception as e:
            logger.error(f"❌ 수화 모델 로드 실패: {e}")
            raise
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        self.window_size = 60  # 2초 (30fps 기준)
        self.pose_buffer = deque(maxlen=self.window_size)
        
        self.decoder = RealtimeDecoder(
            vocab_size=len(self.words),
            confidence_threshold=0.7
        )
        
        # 통계
        self.inference_times = deque(maxlen=100)
        self.detected_words = []
    
    @torch.no_grad()
    def process_pose(self, keypoints: np.ndarray, scores: np.ndarray) -> Optional[str]:
        """포즈 데이터 처리 및 수화 인식"""
        try:
            # 포즈 특징 생성 [133, 3] (x, y, score)
            pose_features = np.zeros((133, 3), dtype=np.float32)
            if keypoints.shape[0] >= 133:
                pose_features[:, :2] = keypoints[:133]
                pose_features[:, 2] = scores[:133] if scores.shape[0] >= 133 else 0.5
            else:
                # 부족한 키포인트는 0으로 패딩
                pose_features[:keypoints.shape[0], :2] = keypoints
                pose_features[:keypoints.shape[0], 2] = scores
            
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
            
            # 프레임 마스크 생성 (실제 프레임만 True)
            actual_frames = min(len(self.pose_buffer), self.window_size)
            frame_masks = torch.zeros(1, self.window_size, dtype=torch.bool, device=self.device)
            frame_masks[0, -actual_frames:] = True  # 마지막 actual_frames만 True
            
            # 모델 추론
            outputs = self.model(input_tensor, frame_masks=frame_masks)
            
            # 현재 프레임의 출력 (마지막 유효한 프레임)
            word_logits = outputs['word_logits'][0, -1]  # [vocab_size]
            boundary_logits = outputs['boundary_logits'][0, -1]  # [3]
            confidence_score = outputs['confidence_scores'][0, -1]  # scalar
            
            # 실시간 디코더 처리
            result = self.decoder.process_frame_output(word_logits, boundary_logits, confidence_score)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            if result is not None:
                word = self.words[result] if result < len(self.words) else f"단어_{result}"
                self.detected_words.append(word)
                logger.info(f"🎯 수화 인식: {word} (신뢰도: {float(confidence_score):.3f})")
                return word
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 수화 추론 실패: {e}")
            return None

class DummyPoseEstimator:
    """더미 포즈 추정기 (개발용)"""
    
    def __init__(self):
        self.device = "cpu"
        logger.info("✅ 더미 포즈 추정기 초기화")
    
    def estimate_pose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """더미 포즈 추정 - 움직이는 사람 시뮬레이션"""
        h, w = image.shape[:2]
        current_time = time.time()
        
        # 시간에 따라 움직이는 키포인트 생성
        keypoints = np.zeros((133, 2))
        scores = np.zeros(133)
        
        center_x = w // 2 + 50 * np.sin(current_time * 0.5)
        center_y = h // 2 + 30 * np.cos(current_time * 0.3)
        
        # 얼굴 (0-67): 중앙 상단
        face_center_x = center_x + 20 * np.sin(current_time * 2)
        face_center_y = center_y - 100
        for i in range(68):
            angle = (i / 68) * 2 * np.pi + current_time * 0.1
            radius = 30 + 10 * np.sin(current_time * 3 + i)
            keypoints[i] = [
                face_center_x + radius * np.cos(angle),
                face_center_y + radius * np.sin(angle)
            ]
            scores[i] = 0.8 + 0.1 * np.sin(current_time * 2 + i)
        
        # 왼손 (68-89): 움직이는 손동작
        left_hand_x = center_x - 150 + 50 * np.sin(current_time * 1.5)
        left_hand_y = center_y + 30 * np.cos(current_time * 1.2)
        for i in range(21):
            finger_angle = (i / 21) * np.pi + current_time
            keypoints[68 + i] = [
                left_hand_x + 30 * np.cos(finger_angle),
                left_hand_y + 30 * np.sin(finger_angle)
            ]
            scores[68 + i] = 0.7 + 0.2 * np.sin(current_time * 4 + i)
        
        # 오른손 (89-110)
        right_hand_x = center_x + 150 + 40 * np.cos(current_time * 1.8)
        right_hand_y = center_y + 20 * np.sin(current_time * 1.5)
        for i in range(21):
            finger_angle = (i / 21) * np.pi - current_time
            keypoints[89 + i] = [
                right_hand_x + 25 * np.cos(finger_angle),
                right_hand_y + 25 * np.sin(finger_angle)
            ]
            scores[89 + i] = 0.6 + 0.3 * np.cos(current_time * 3 + i)
        
        # 몸 키포인트 (110-133)
        body_positions = [
            [center_x, center_y - 50],  # 목
            [center_x - 60, center_y - 30],  # 왼쪽 어깨
            [center_x + 60, center_y - 30],  # 오른쪽 어깨
            [center_x, center_y],  # 가슴 중앙
        ]
        
        for i in range(min(23, len(body_positions))):
            if i < len(body_positions):
                keypoints[110 + i] = body_positions[i]
            else:
                keypoints[110 + i] = [center_x, center_y]
            scores[110 + i] = 0.9
        
        # 나머지 키포인트는 기본값
        for i in range(110 + len(body_positions), 133):
            keypoints[i] = [center_x, center_y]
            scores[i] = 0.5
        
        return keypoints, scores

class RealMosuServer:
    """실제 수화 모델을 사용하는 MOSU 서버"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 host: str = "0.0.0.0",
                 port: int = 8002):
        
        self.host = host
        self.port = port
        
        # 컴포넌트 초기화
        self.pose_estimator = DummyPoseEstimator()  # RTMW 대신 더미 사용
        
        try:
            self.sign_inferencer = RealSignLanguageInferencer(model_path, device)
        except Exception as e:
            logger.error(f"❌ 실제 수화 모델 로딩 실패: {e}")
            logger.info("🔄 더미 모델로 폴백합니다")
            from simple_server import DummySignLanguageInferencer
            self.sign_inferencer = DummySignLanguageInferencer()
        
        # FastAPI 앱 설정
        self.app = FastAPI(title="MOSU Real Sign Language Server")
        
        # WebSocket 연결 관리
        self.connections: List[WebSocket] = []
        
        self.setup_routes()
        
        logger.info(f"🚀 실제 MOSU 서버 초기화 완료")
        logger.info(f"   - 서버 주소: {host}:{port}")
        logger.info(f"   - 모델 타입: {'실제' if hasattr(self.sign_inferencer, 'model') else '더미'}")
    
    def setup_routes(self):
        """라우트 설정"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """메인 페이지"""
            return self.get_advanced_html()
        
        @self.app.get("/health")
        async def health():
            """헬스 체크"""
            model_type = "real" if hasattr(self.sign_inferencer, 'model') else "dummy"
            vocab_size = len(getattr(self.sign_inferencer, 'words', []))
            
            return {
                "status": "healthy",
                "model_type": model_type,
                "vocab_size": vocab_size,
                "connections": len(self.connections),
                "device": getattr(self.sign_inferencer, 'device', 'cpu')
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """상세 통계"""
            inference_times = getattr(self.sign_inferencer, 'inference_times', [])
            detected_words = getattr(self.sign_inferencer, 'detected_words', [])
            
            return {
                "connections": len(self.connections),
                "detected_words": detected_words[-20:],  # 최근 20개
                "inference": {
                    "count": len(inference_times),
                    "avg_time": float(np.mean(inference_times)) if inference_times else 0,
                    "min_time": float(np.min(inference_times)) if inference_times else 0,
                    "max_time": float(np.max(inference_times)) if inference_times else 0
                },
                "model_info": {
                    "type": "real" if hasattr(self.sign_inferencer, 'model') else "dummy",
                    "vocab_size": len(getattr(self.sign_inferencer, 'words', [])),
                    "device": str(getattr(self.sign_inferencer, 'device', 'cpu'))
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
                    data = await websocket.receive_json()
                    
                    if data["type"] == "frame":
                        try:
                            # Base64 이미지 디코딩
                            image_data = base64.b64decode(data["image"].split(",")[1])
                            image = Image.open(io.BytesIO(image_data))
                            image_np = np.array(image)
                            
                            # RGB to BGR
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
                                    "confidence": float(np.mean(scores)) if len(scores) > 0 else 0.0
                                },
                                "stats": {
                                    "avg_inference_time": float(np.mean(getattr(self.sign_inferencer, 'inference_times', [0]))) if hasattr(self.sign_inferencer, 'inference_times') else 0
                                }
                            }
                            
                            await websocket.send_json(response)
                            
                        except Exception as e:
                            logger.error(f"❌ 프레임 처리 오류: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
                            
            except WebSocketDisconnect:
                if websocket in self.connections:
                    self.connections.remove(websocket)
                logger.info(f"🔌 연결 종료: 총 {len(self.connections)}개")
            except Exception as e:
                logger.error(f"❌ WebSocket 오류: {e}")
                if websocket in self.connections:
                    self.connections.remove(websocket)
    
    def get_advanced_html(self) -> str:
        """고급 웹 인터페이스"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MOSU 실시간 수화 인식 - 실제 모델</title>
            <meta charset="UTF-8">
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333; min-height: 100vh;
                }
                .container { 
                    max-width: 1400px; margin: 0 auto; 
                    background: rgba(255,255,255,0.95); 
                    border-radius: 15px; padding: 30px; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }
                .header { 
                    text-align: center; margin-bottom: 30px; 
                    background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                }
                .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
                .video-section { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
                .video-box { position: relative; }
                video, canvas { 
                    width: 100%; max-width: 100%; height: auto; 
                    border: 3px solid #4facfe; border-radius: 10px; 
                    background: #f8f9fa;
                }
                .controls { 
                    text-align: center; margin: 20px 0; 
                    display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
                }
                button { 
                    padding: 12px 24px; font-size: 16px; font-weight: 600;
                    border: none; border-radius: 25px; cursor: pointer; 
                    transition: all 0.3s ease; text-transform: uppercase;
                }
                .btn-start { background: linear-gradient(45deg, #4facfe, #00f2fe); color: white; }
                .btn-stop { background: linear-gradient(45deg, #fa709a, #fee140); color: white; }
                .btn-clear { background: linear-gradient(45deg, #a8edea, #fed6e3); color: #333; }
                button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }
                
                .results-section { 
                    background: white; padding: 25px; border-radius: 10px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }
                .current-word { 
                    font-size: 36px; font-weight: bold; text-align: center;
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    margin: 20px 0; min-height: 50px;
                    display: flex; align-items: center; justify-content: center;
                    border: 2px dashed #ddd; border-radius: 10px; padding: 20px;
                }
                
                .stats-grid { 
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                    gap: 15px; margin: 20px 0;
                }
                .stat-card { 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 15px; border-radius: 10px; text-align: center;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
                }
                .stat-value { font-size: 24px; font-weight: bold; }
                .stat-label { font-size: 12px; opacity: 0.9; text-transform: uppercase; }
                
                .detected-words { 
                    background: #f8f9fa; padding: 20px; border-radius: 10px; 
                    min-height: 120px; border: 1px solid #e9ecef;
                    max-height: 200px; overflow-y: auto;
                }
                .word-tag { 
                    display: inline-block; margin: 5px; padding: 8px 16px; 
                    background: linear-gradient(45deg, #4facfe, #00f2fe); 
                    color: white; border-radius: 20px; font-weight: 600;
                    animation: fadeIn 0.5s ease-in;
                }
                @keyframes fadeIn { from { opacity: 0; transform: scale(0.8); } to { opacity: 1; transform: scale(1); } }
                
                .status-indicator { 
                    display: inline-block; width: 12px; height: 12px; 
                    border-radius: 50%; margin-right: 8px;
                }
                .status-connected { background: #28a745; }
                .status-disconnected { background: #dc3545; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤲 MOSU 실시간 수화 인식</h1>
                    <p>실제 Transformer 모델을 사용한 고성능 수화 인식 시스템</p>
                </div>
                
                <div class="main-content">
                    <div class="video-section">
                        <h3>📹 웹캠 입력</h3>
                        <div class="video-box">
                            <video id="video" autoplay muted></video>
                        </div>
                        
                        <h3>🎯 포즈 추정</h3>
                        <div class="video-box">
                            <canvas id="canvas"></canvas>
                        </div>
                        
                        <div class="controls">
                            <button class="btn-start" onclick="startSystem()">🚀 시스템 시작</button>
                            <button class="btn-stop" onclick="stopSystem()">⏹️ 정지</button>
                            <button class="btn-clear" onclick="clearResults()">🗑️ 결과 초기화</button>
                        </div>
                    </div>
                    
                    <div class="results-section">
                        <h3>🎯 실시간 인식 결과</h3>
                        <div class="current-word" id="current-word">시스템 준비 중...</div>
                        
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value" id="connection-status">
                                    <span class="status-indicator status-disconnected"></span>
                                    대기 중
                                </div>
                                <div class="stat-label">연결 상태</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="fps-counter">0</div>
                                <div class="stat-label">FPS</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="confidence-level">0%</div>
                                <div class="stat-label">신뢰도</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" id="inference-time">0ms</div>
                                <div class="stat-label">추론 시간</div>
                            </div>
                        </div>
                        
                        <h4>📝 감지된 단어들</h4>
                        <div class="detected-words" id="detected-words">
                            <p style="text-align: center; color: #6c757d; margin: 50px 0;">
                                아직 감지된 단어가 없습니다.<br>
                                손으로 수화를 해보세요!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let video = document.getElementById('video');
                let canvas = document.getElementById('canvas');
                let ctx = canvas.getContext('2d');
                let ws = null;
                let stream = null;
                let isRunning = false;
                let frameCount = 0;
                let lastFpsTime = Date.now();
                let detectedWords = new Set();
                
                async function startSystem() {
                    try {
                        // 웹캠 시작
                        stream = await navigator.mediaDevices.getUserMedia({
                            video: { width: 640, height: 480, frameRate: 30 }
                        });
                        video.srcObject = stream;
                        
                        canvas.width = 640;
                        canvas.height = 480;
                        
                        // WebSocket 연결
                        connectWebSocket();
                        
                        document.getElementById('current-word').textContent = '시스템 시작됨 - 수화를 해보세요!';
                        
                    } catch (error) {
                        console.error('시스템 시작 오류:', error);
                        alert('웹캠에 접근할 수 없습니다: ' + error.message);
                    }
                }
                
                function connectWebSocket() {
                    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(`${protocol}//${location.host}/ws`);
                    
                    ws.onopen = function() {
                        updateConnectionStatus(true);
                        isRunning = true;
                        console.log('WebSocket 연결 성공');
                        sendFrames();
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'result') {
                            handleResult(data);
                        } else if (data.type === 'error') {
                            console.error('서버 오류:', data.message);
                        }
                    };
                    
                    ws.onclose = function() {
                        updateConnectionStatus(false);
                        isRunning = false;
                        console.log('WebSocket 연결 종료');
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket 오류:', error);
                        updateConnectionStatus(false);
                    };
                }
                
                function handleResult(data) {
                    // 포즈 시각화
                    drawPose(data.pose);
                    
                    // 수화 결과 표시
                    if (data.sign.word) {
                        document.getElementById('current-word').textContent = data.sign.word;
                        addDetectedWord(data.sign.word);
                    }
                    
                    // 통계 업데이트
                    const confidence = Math.round(data.sign.confidence * 100);
                    document.getElementById('confidence-level').textContent = confidence + '%';
                    
                    if (data.stats && data.stats.avg_inference_time) {
                        const inferenceMs = Math.round(data.stats.avg_inference_time * 1000);
                        document.getElementById('inference-time').textContent = inferenceMs + 'ms';
                    }
                    
                    // FPS 계산
                    frameCount++;
                    const now = Date.now();
                    if (now - lastFpsTime > 1000) {
                        const fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
                        document.getElementById('fps-counter').textContent = fps;
                        frameCount = 0;
                        lastFpsTime = now;
                    }
                }
                
                function sendFrames() {
                    if (!isRunning || !ws || ws.readyState !== WebSocket.OPEN) {
                        return;
                    }
                    
                    try {
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = 640;
                        tempCanvas.height = 480;
                        const tempCtx = tempCanvas.getContext('2d');
                        tempCtx.drawImage(video, 0, 0, 640, 480);
                        
                        const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);
                        
                        ws.send(JSON.stringify({
                            type: 'frame',
                            image: dataUrl,
                            timestamp: Date.now()
                        }));
                        
                    } catch (error) {
                        console.error('프레임 전송 오류:', error);
                    }
                    
                    setTimeout(sendFrames, 100); // 10 FPS
                }
                
                function drawPose(pose) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    const keypoints = pose.keypoints;
                    const scores = pose.scores;
                    
                    // 키포인트 그리기
                    for (let i = 0; i < keypoints.length && i < scores.length; i++) {
                        if (scores[i] > 0.5) {
                            const x = keypoints[i][0] * (canvas.width / video.videoWidth);
                            const y = keypoints[i][1] * (canvas.height / video.videoHeight);
                            
                            // 다른 신체 부위별로 다른 색상
                            if (i < 68) ctx.fillStyle = '#ff6b6b'; // 얼굴: 빨강
                            else if (i < 89) ctx.fillStyle = '#4ecdc4'; // 왼손: 청록
                            else if (i < 110) ctx.fillStyle = '#45b7d1'; // 오른손: 파랑
                            else ctx.fillStyle = '#96ceb4'; // 몸: 초록
                            
                            ctx.beginPath();
                            ctx.arc(x, y, 3, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    }
                }
                
                function addDetectedWord(word) {
                    if (!detectedWords.has(word)) {
                        detectedWords.add(word);
                        updateDetectedWordsDisplay();
                    }
                }
                
                function updateDetectedWordsDisplay() {
                    const container = document.getElementById('detected-words');
                    if (detectedWords.size === 0) {
                        container.innerHTML = `
                            <p style="text-align: center; color: #6c757d; margin: 50px 0;">
                                아직 감지된 단어가 없습니다.<br>
                                손으로 수화를 해보세요!
                            </p>`;
                    } else {
                        container.innerHTML = Array.from(detectedWords).map(word => 
                            `<span class="word-tag">${word}</span>`
                        ).join('');
                    }
                }
                
                function updateConnectionStatus(connected) {
                    const statusEl = document.getElementById('connection-status');
                    if (connected) {
                        statusEl.innerHTML = '<span class="status-indicator status-connected"></span>연결됨';
                    } else {
                        statusEl.innerHTML = '<span class="status-indicator status-disconnected"></span>연결 끊김';
                    }
                }
                
                function stopSystem() {
                    isRunning = false;
                    
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                        stream = null;
                    }
                    
                    if (ws) {
                        ws.close();
                    }
                    
                    updateConnectionStatus(false);
                    document.getElementById('current-word').textContent = '시스템 정지됨';
                    document.getElementById('fps-counter').textContent = '0';
                    
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
                
                function clearResults() {
                    detectedWords.clear();
                    updateDetectedWordsDisplay();
                    document.getElementById('confidence-level').textContent = '0%';
                    document.getElementById('inference-time').textContent = '0ms';
                }
                
                // 페이지 종료 시 정리
                window.addEventListener('beforeunload', stopSystem);
            </script>
        </body>
        </html>
        """
    
    def run(self):
        """서버 실행"""
        model_type = "실제" if hasattr(self.sign_inferencer, 'model') else "더미"
        logger.info(f"\n🚀 MOSU 서버 시작 ({model_type} 모델)")
        logger.info(f"   - 주소: http://{self.host}:{self.port}")
        logger.info(f"   - 모델: {model_type}")
        logger.info(f"   - Ctrl+C로 종료")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("\n⏹️ 서버 종료")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MOSU Real Sign Language Server")
    parser.add_argument("--model", type=str, 
                       default="../mosumodel/best_model_stage_1.pt",
                       help="수화 인식 모델 경로")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "xpu"],
                       help="추론 디바이스")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8002, help="서버 포트")
    
    args = parser.parse_args()
    
    try:
        server = RealMosuServer(
            model_path=args.model,
            device=args.device,
            host=args.host,
            port=args.port
        )
        server.run()
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
