#!/usr/bin/env python3
"""
MOSU 통합 백엔드 서버
- RTMW 포즈 추정 통합
- 수화 인식 처리
- 웹소켓 실시간 통신
192.168.100.26:8000에서 실행
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
import sys
import os

# FastAPI & WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# 프로젝트 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class DummyPoseEstimator:
    """더미 포즈 추정기 (MMPose 없이도 동작)"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        logger.info("✅ 더미 포즈 추정기 초기화")
    
    def estimate_pose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """더미 포즈 추정 (133개 키포인트)"""
        h, w = image.shape[:2]
        
        # 133개 더미 키포인트 생성 (RTMW 형식)
        keypoints = np.random.rand(133, 2) * np.array([w, h])
        scores = np.random.rand(133) * 0.7 + 0.2  # 0.2-0.9 범위
        
        # 더 현실적인 포즈 생성
        center_x, center_y = w // 2, h // 2
        
        # 얼굴 영역 (0-68)
        keypoints[:68] = np.random.normal([center_x, center_y * 0.3], [w*0.1, h*0.1], (68, 2))
        
        # 몸통 영역 (68-91) 
        keypoints[68:91] = np.random.normal([center_x, center_y], [w*0.15, h*0.2], (23, 2))
        
        # 왼손 (91-112)
        keypoints[91:112] = np.random.normal([center_x - w*0.2, center_y], [w*0.05, h*0.1], (21, 2))
        
        # 오른손 (112-133)
        keypoints[112:133] = np.random.normal([center_x + w*0.2, center_y], [w*0.05, h*0.1], (21, 2))
        
        # 경계 체크
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, w)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, h)
        
        return keypoints.astype(np.float32), scores.astype(np.float32)

class DummySignLanguageInferencer:
    """더미 수화 인식 추론기"""
    
    def __init__(self):
        self.words = [
            "안녕하세요", "감사합니다", "죄송합니다", "네", "아니요",
            "좋아요", "괜찮아요", "미안해요", "도와주세요", "사랑해요",
            "만나서반가워요", "안녕히가세요", "고맙습니다", "반갑습니다", 
            "수고하세요", "화이팅", "축하해요", "생일축하해요", "새해복많이받으세요", "건강하세요"
        ]
        
        self.window_size = 60  # 2초 (30fps)
        self.pose_buffer = deque(maxlen=self.window_size)
        self.inference_times = deque(maxlen=100)
        
        logger.info(f"✅ 더미 수화 인식기 초기화 (단어: {len(self.words)}개)")
    
    @torch.no_grad()
    def process_pose(self, keypoints: np.ndarray, scores: np.ndarray) -> Optional[str]:
        """포즈 데이터 처리 및 더미 수화 인식"""
        # 포즈 특징 생성
        pose_features = np.zeros((133, 3), dtype=np.float32)
        pose_features[:, :2] = keypoints
        pose_features[:, 2] = scores
        
        # 버퍼에 추가
        self.pose_buffer.append(pose_features)
        
        # 최소 프레임 수 확인 (1초)
        if len(self.pose_buffer) < 30:
            return None
        
        start_time = time.time()
        
        # 더미 추론 (손 움직임 기반 간단한 패턴 매칭)
        recent_frames = list(self.pose_buffer)[-30:]  # 최근 1초
        
        # 손 영역의 움직임 계산
        hand_movement = 0
        if len(recent_frames) > 1:
            for i in range(1, len(recent_frames)):
                prev_hands = np.concatenate([recent_frames[i-1][91:112], recent_frames[i-1][112:133]])
                curr_hands = np.concatenate([recent_frames[i][91:112], recent_frames[i][112:133]])
                movement = np.mean(np.linalg.norm(curr_hands - prev_hands, axis=1))
                hand_movement += movement
        
        # 움직임 강도에 따른 단어 선택
        if hand_movement > 50:  # 활발한 움직임
            word_idx = int((hand_movement * 7) % len(self.words))
            selected_word = self.words[word_idx]
            
            self.inference_times.append(time.time() - start_time)
            return selected_word
        
        return None

class FullMosuServer:
    """통합 MOSU 서버 (포즈 추정 + 수화 인식)"""
    
    def __init__(self, 
                 device: str = "auto",
                 host: str = "0.0.0.0",
                 port: int = 8000):
        
        self.host = host
        self.port = port
        self.device = self._determine_device(device)
        
        # 컴포넌트 초기화
        self.pose_estimator = DummyPoseEstimator(device=self.device)
        self.sign_inferencer = DummySignLanguageInferencer()
        
        # FastAPI 앱 설정
        self.app = FastAPI(title="MOSU Full Backend Server")
        
        # WebSocket 연결 관리
        self.connections: List[WebSocket] = []
        
        # 통계
        self.frame_count = 0
        self.word_count = 0
        self.start_time = time.time()
        
        self.setup_routes()
        
        logger.info(f"🚀 MOSU 통합 서버 초기화 완료")
        logger.info(f"   - 서버 주소: {host}:{port}")
        logger.info(f"   - 디바이스: {self.device}")
        logger.info(f"   - 단어 수: {len(self.sign_inferencer.words)}개")
    
    def _determine_device(self, device: str) -> str:
        """디바이스 자동 결정"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                return "xpu"
            else:
                return "cpu"
        return device
    
    def setup_routes(self):
        """라우트 설정"""
        
        @self.app.get("/health")
        async def health_check():
            """헬스체크"""
            uptime = time.time() - self.start_time
            return {
                "status": "healthy",
                "uptime": f"{uptime:.1f}s",
                "device": self.device,
                "frame_count": self.frame_count,
                "word_count": self.word_count,
                "fps": self.frame_count / uptime if uptime > 0 else 0
            }
        
        @self.app.get("/stats")
        async def get_stats():
            """통계 정보"""
            uptime = time.time() - self.start_time
            avg_inference_time = 0
            if self.sign_inferencer.inference_times:
                avg_inference_time = np.mean(self.sign_inferencer.inference_times)
            
            return {
                "uptime": uptime,
                "frame_count": self.frame_count,
                "word_count": self.word_count,
                "fps": self.frame_count / uptime if uptime > 0 else 0,
                "avg_inference_time": avg_inference_time,
                "connected_clients": len(self.connections),
                "buffer_size": len(self.sign_inferencer.pose_buffer),
                "vocab_size": len(self.sign_inferencer.words)
            }
        
        @self.app.post("/estimate_pose")
        async def estimate_pose_endpoint(
            image: UploadFile = File(...),
            frame_id: str = Form(default="0"),
            bbox: str = Form(default="[]"),
            timestamp: str = Form(default="0")
        ):
            """포즈 추정 엔드포인트 (기존 pose-server 호환)"""
            try:
                # 이미지 읽기
                image_bytes = await image.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise HTTPException(status_code=400, detail="Invalid image")
                
                # 포즈 추정
                keypoints, scores = self.pose_estimator.estimate_pose(img)
                
                return {
                    "status": "success",
                    "frame_id": frame_id,
                    "keypoints": keypoints.tolist(),
                    "scores": scores.tolist(),
                    "timestamp": float(timestamp),
                    "processing_time": 0.05  # 더미 처리 시간
                }
                
            except Exception as e:
                logger.error(f"❌ 포즈 추정 오류: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 엔드포인트"""
            await websocket.accept()
            self.connections.append(websocket)
            logger.info(f"🔗 새 클라이언트 연결: {len(self.connections)}개")
            
            try:
                while True:
                    data = await websocket.receive_json()
                    
                    if data['type'] == 'frame':
                        # Base64 이미지 디코딩
                        image_data = base64.b64decode(data['data'])
                        nparr = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            self.frame_count += 1
                            
                            # 포즈 추정
                            keypoints, scores = self.pose_estimator.estimate_pose(image)
                            
                            # 수화 인식
                            word = self.sign_inferencer.process_pose(keypoints, scores)
                            
                            if word:
                                self.word_count += 1
                                # 클라이언트에 결과 전송
                                response = {
                                    'type': 'result',
                                    'word': word,
                                    'confidence': np.random.random() * 0.3 + 0.7,  # 0.7-1.0 더미 신뢰도
                                    'timestamp': time.time(),
                                    'frame_id': self.frame_count,
                                    'keypoints_count': len(keypoints)
                                }
                                
                                await websocket.send_json(response)
                            
                            # 주기적으로 상태 전송 (10프레임마다)
                            if self.frame_count % 10 == 0:
                                uptime = time.time() - self.start_time
                                status_response = {
                                    'type': 'status',
                                    'fps': self.frame_count / uptime if uptime > 0 else 0,
                                    'word_count': self.word_count,
                                    'buffer_size': len(self.sign_inferencer.pose_buffer)
                                }
                                await websocket.send_json(status_response)
                    
            except WebSocketDisconnect:
                self.connections.remove(websocket)
                logger.info(f"📡 클라이언트 연결 해제: {len(self.connections)}개")
            except Exception as e:
                logger.error(f"❌ WebSocket 오류: {e}")
                if websocket in self.connections:
                    self.connections.remove(websocket)
    
    def run(self):
        """서버 실행"""
        logger.info(f"\n🚀 MOSU 통합 백엔드 서버 시작")
        logger.info(f"   - 주소: http://{self.host}:{self.port}")
        logger.info(f"   - 헬스체크: http://{self.host}:{self.port}/health")
        logger.info(f"   - 통계: http://{self.host}:{self.port}/stats")
        logger.info(f"   - WebSocket: ws://{self.host}:{self.port}/ws")
        logger.info(f"   - 포즈 추정: POST /estimate_pose")
        logger.info(f"   - Ctrl+C로 종료")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("\n⏹️ MOSU 통합 서버 종료")
        except Exception as e:
            logger.error(f"❌ 서버 실행 실패: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MOSU Full Backend Server")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "xpu"],
                       help="추론 디바이스")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000,
                       help="서버 포트")
    
    args = parser.parse_args()
    
    try:
        server = FullMosuServer(
            device=args.device,
            host=args.host,
            port=args.port
        )
        
        server.run()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ 서버 종료")
    except Exception as e:
        logger.error(f"❌ 서버 실행 실패: {e}")

if __name__ == "__main__":
    main()
