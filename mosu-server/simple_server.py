#!/usr/bin/env python3
"""
개발용 간단한 MOSU 서버 (RTMW 없이)
MMPose 의존성 없이 테스트할 수 있는 버전
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

class DummySignLanguageInferencer:
    """개발용 더미 수화 인식기"""
    
    def __init__(self):
        self.words = [
            "안녕하세요", "감사합니다", "미안합니다", "사랑해요", "괜찮아요",
            "도와주세요", "좋아요", "싫어요", "네", "아니요",
            "먹다", "마시다", "가다", "오다", "보다",
            "듣다", "말하다", "웃다", "울다", "자다"
        ]
        self.word_index = 0
        self.frame_count = 0
        self.detected_words = []
        
        logger.info(f"✅ 더미 수화 인식기 초기화 (단어: {len(self.words)}개)")
    
    def process_pose(self, keypoints: np.ndarray, scores: np.ndarray) -> Optional[str]:
        """더미 수화 인식"""
        self.frame_count += 1
        
        # 60프레임마다 랜덤하게 단어 감지
        if self.frame_count % 60 == 0:
            if np.random.random() > 0.3:  # 70% 확률로 단어 감지
                word = self.words[self.word_index % len(self.words)]
                self.word_index += 1
                self.detected_words.append(word)
                logger.info(f"🎯 더미 수화 인식: {word}")
                return word
        
        return None

class SimplePoseEstimator:
    """간단한 포즈 추정기 (더미)"""
    
    def __init__(self):
        self.device = "cpu"
        logger.info("✅ 더미 포즈 추정기 초기화")
    
    def estimate_pose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """더미 포즈 추정"""
        h, w = image.shape[:2]
        
        # 더미 키포인트 생성 (133개)
        # 사람 형태로 더 그럴듯한 키포인트 생성
        keypoints = np.zeros((133, 2))
        scores = np.zeros(133)
        
        # 기본적인 인체 구조 시뮬레이션
        center_x, center_y = w // 2, h // 2
        
        # 얼굴 (0-67)
        face_keypoints = self._generate_face_keypoints(center_x, center_y - 100, w, h)
        keypoints[:68] = face_keypoints
        scores[:68] = np.random.uniform(0.7, 0.95, 68)
        
        # 손 (68-110, 111-133)
        left_hand = self._generate_hand_keypoints(center_x - 150, center_y + 50)
        right_hand = self._generate_hand_keypoints(center_x + 150, center_y + 50)
        
        keypoints[68:89] = left_hand  # 왼손
        keypoints[89:110] = right_hand  # 오른손
        scores[68:110] = np.random.uniform(0.6, 0.9, 42)
        
        # 몸 키포인트 (111-133)
        body_keypoints = self._generate_body_keypoints(center_x, center_y)
        keypoints[110:133] = body_keypoints
        scores[110:133] = np.random.uniform(0.8, 0.95, 23)
        
        # 약간의 움직임 추가
        keypoints += np.random.normal(0, 2, keypoints.shape)
        
        return keypoints, scores
    
    def _generate_face_keypoints(self, cx, cy, w, h):
        """얼굴 키포인트 생성"""
        face_points = []
        
        # 얼굴 윤곽
        for i in range(17):
            angle = (i - 8) * 0.2
            x = cx + 60 * np.sin(angle)
            y = cy + 80 + 20 * np.cos(angle)
            face_points.append([x, y])
        
        # 눈썹, 눈, 코, 입 (더미)
        for i in range(51):
            x = cx + np.random.uniform(-50, 50)
            y = cy + np.random.uniform(-30, 50)
            face_points.append([x, y])
        
        return np.array(face_points)
    
    def _generate_hand_keypoints(self, cx, cy):
        """손 키포인트 생성"""
        hand_points = []
        
        # 손목
        hand_points.append([cx, cy])
        
        # 손가락 (5개 * 4관절)
        for finger in range(5):
            finger_angle = (finger - 2) * 0.4
            for joint in range(4):
                x = cx + (joint + 1) * 15 * np.cos(finger_angle)
                y = cy - (joint + 1) * 15 * np.sin(finger_angle)
                hand_points.append([x, y])
        
        return np.array(hand_points)
    
    def _generate_body_keypoints(self, cx, cy):
        """몸 키포인트 생성"""
        body_points = [
            [cx, cy - 200],      # 머리
            [cx, cy - 150],      # 목
            [cx - 80, cy - 100], # 왼쪽 어깨
            [cx + 80, cy - 100], # 오른쪽 어깨
            [cx - 100, cy],      # 왼쪽 팔꿈치
            [cx + 100, cy],      # 오른쪽 팔꿈치
            [cx - 120, cy + 80], # 왼쪽 손목
            [cx + 120, cy + 80], # 오른쪽 손목
            [cx, cy - 50],       # 가슴
            [cx - 40, cy + 100], # 왼쪽 엉덩이
            [cx + 40, cy + 100], # 오른쪽 엉덩이
            [cx - 50, cy + 200], # 왼쪽 무릎
            [cx + 50, cy + 200], # 오른쪽 무릎
            [cx - 60, cy + 300], # 왼쪽 발목
            [cx + 60, cy + 300], # 오른쪽 발목
        ]
        
        # 부족한 포인트는 더미로 채우기
        while len(body_points) < 23:
            body_points.append([cx, cy])
        
        return np.array(body_points[:23])

class SimpleMosuServer:
    """간단한 MOSU 서버 (개발용)"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        
        # 컴포넌트 초기화
        self.pose_estimator = SimplePoseEstimator()
        
        if MODEL_AVAILABLE:
            # 실제 모델이 있다면 더미 대신 실제 모델 사용
            try:
                model_path = Path("../mosumodel/best_model_stage_1.pt")
                if model_path.exists():
                    logger.info("🔍 실제 수화 모델 로딩 시도...")
                    # 실제 모델 로딩 로직은 여기에 추가
                    pass
            except Exception as e:
                logger.warning(f"⚠️ 실제 모델 로딩 실패: {e}")
        
        self.sign_inferencer = DummySignLanguageInferencer()
        
        # FastAPI 앱 설정
        self.app = FastAPI(title="Simple MOSU Server")
        
        # 정적 파일 서빙 (선택적)
        try:
            self.app.mount("/static", StaticFiles(directory="static"), name="static")
        except:
            pass
        
        # WebSocket 연결 관리
        self.connections: List[WebSocket] = []
        
        self.setup_routes()
        
        logger.info(f"🚀 간단한 MOSU 서버 초기화 완료")
        logger.info(f"   - 서버 주소: {host}:{port}")
    
    def setup_routes(self):
        """라우트 설정"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """메인 페이지"""
            return self.get_simple_html()
        
        @self.app.get("/health")
        async def health():
            """헬스 체크"""
            return {
                "status": "healthy",
                "model_type": "dummy" if not MODEL_AVAILABLE else "loaded",
                "connections": len(self.connections)
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 엔드포인트"""
            await websocket.accept()
            self.connections.append(websocket)
            logger.info(f"🔗 새 연결: 총 {len(self.connections)}개")
            
            try:
                while True:
                    # 클라이언트로부터 데이터 수신
                    data = await websocket.receive_json()
                    
                    if data["type"] == "frame":
                        # Base64 이미지 디코딩
                        try:
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
    
    def get_simple_html(self) -> str:
        """간단한 테스트 페이지"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Simple MOSU 테스트</title>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                video, canvas { width: 320px; height: 240px; border: 1px solid #ddd; margin: 10px; }
                button { padding: 10px 20px; margin: 5px; font-size: 16px; }
                .result { background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .word { font-size: 24px; font-weight: bold; color: #0066cc; }
                .status { padding: 10px; background: #e8f5e8; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Simple MOSU 테스트</h1>
                
                <div>
                    <video id="video" autoplay muted></video>
                    <canvas id="canvas"></canvas>
                </div>
                
                <div>
                    <button onclick="startTest()">📹 테스트 시작</button>
                    <button onclick="stopTest()">⏹️ 정지</button>
                </div>
                
                <div class="status" id="status">대기 중...</div>
                
                <div class="result">
                    <div class="word" id="current-word">-</div>
                    <div>신뢰도: <span id="confidence">0%</span></div>
                    <div>FPS: <span id="fps">0</span></div>
                </div>
                
                <div>
                    <h3>감지된 단어들:</h3>
                    <div id="detected-words"></div>
                </div>
            </div>
            
            <script>
                let video = document.getElementById('video');
                let canvas = document.getElementById('canvas');
                let ctx = canvas.getContext('2d');
                let ws = null;
                let stream = null;
                let running = false;
                
                function startTest() {
                    navigator.mediaDevices.getUserMedia({ video: true })
                        .then(function(mediaStream) {
                            stream = mediaStream;
                            video.srcObject = stream;
                            canvas.width = 320;
                            canvas.height = 240;
                            
                            // WebSocket 연결
                            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                            ws = new WebSocket(`${protocol}//${location.host}/ws`);
                            
                            ws.onopen = function() {
                                document.getElementById('status').textContent = '✅ 연결됨';
                                running = true;
                                sendFrames();
                            };
                            
                            ws.onmessage = function(event) {
                                const data = JSON.parse(event.data);
                                
                                if (data.type === 'result') {
                                    // 포즈 시각화
                                    drawPose(data.pose);
                                    
                                    // 결과 표시
                                    if (data.sign.word) {
                                        document.getElementById('current-word').textContent = data.sign.word;
                                        addWord(data.sign.word);
                                    }
                                    
                                    const confidence = Math.round(data.sign.confidence * 100);
                                    document.getElementById('confidence').textContent = confidence + '%';
                                }
                            };
                            
                            ws.onclose = function() {
                                document.getElementById('status').textContent = '❌ 연결 종료';
                                running = false;
                            };
                        })
                        .catch(function(error) {
                            alert('카메라 접근 실패: ' + error.message);
                        });
                }
                
                function stopTest() {
                    running = false;
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                    if (ws) {
                        ws.close();
                    }
                    document.getElementById('status').textContent = '⏹️ 정지됨';
                }
                
                function sendFrames() {
                    if (!running || !ws || ws.readyState !== WebSocket.OPEN) {
                        return;
                    }
                    
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = 320;
                    tempCanvas.height = 240;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(video, 0, 0, 320, 240);
                    
                    const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);
                    
                    ws.send(JSON.stringify({
                        type: 'frame',
                        image: dataUrl
                    }));
                    
                    setTimeout(sendFrames, 200); // 5 FPS
                }
                
                function drawPose(pose) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    const keypoints = pose.keypoints;
                    const scores = pose.scores;
                    
                    ctx.fillStyle = 'red';
                    for (let i = 0; i < keypoints.length && i < scores.length; i++) {
                        if (scores[i] > 0.5) {
                            const x = keypoints[i][0] * (canvas.width / video.videoWidth);
                            const y = keypoints[i][1] * (canvas.height / video.videoHeight);
                            ctx.beginPath();
                            ctx.arc(x, y, 2, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    }
                }
                
                let detectedWords = [];
                function addWord(word) {
                    if (!detectedWords.includes(word)) {
                        detectedWords.push(word);
                        const container = document.getElementById('detected-words');
                        container.innerHTML = detectedWords.map(w => 
                            `<span style="margin: 3px; padding: 5px; background: #0066cc; color: white; border-radius: 10px; display: inline-block;">${w}</span>`
                        ).join('');
                    }
                }
            </script>
        </body>
        </html>
        """
    
    def run(self):
        """서버 실행"""
        logger.info(f"\n🚀 간단한 MOSU 서버 시작")
        logger.info(f"   - 주소: http://{self.host}:{self.port}")
        logger.info(f"   - 모델: {'더미' if not MODEL_AVAILABLE else '실제'}")
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
    
    parser = argparse.ArgumentParser(description="Simple MOSU Server for Development")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    
    args = parser.parse_args()
    
    try:
        server = SimpleMosuServer(host=args.host, port=args.port)
        server.run()
    except Exception as e:
        logger.error(f"❌ 서버 시작 실패: {e}")

if __name__ == "__main__":
    main()
