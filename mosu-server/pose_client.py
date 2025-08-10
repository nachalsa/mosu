#!/usr/bin/env python3
"""
포즈 서버 클라이언트 - MOSU 서버에서 pose-server와 통신
"""

import cv2
import numpy as np
import requests
import json
import time
import logging
from typing import Optional, Tuple
from pathlib import Path
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

class PoseServerClient:
    """포즈 서버 클라이언트"""
    
    def __init__(self, pose_server_url: str = "http://192.168.100.135:5000"):
        self.pose_server_url = pose_server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 5  # 5초 타임아웃
        
        # 연결 테스트
        self._test_connection()
        
        logger.info(f"✅ 포즈 서버 클라이언트 초기화: {self.pose_server_url}")
    
    def _test_connection(self):
        """포즈 서버 연결 테스트"""
        try:
            response = self.session.get(f"{self.pose_server_url}/health")
            if response.status_code == 200:
                logger.info(f"✅ 포즈 서버 연결 성공: {self.pose_server_url}")
                data = response.json()
                logger.info(f"   - 상태: {data.get('status', 'unknown')}")
                logger.info(f"   - 디바이스: {data.get('device', 'unknown')}")
            else:
                logger.warning(f"⚠️ 포즈 서버 응답 오류: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 포즈 서버 연결 실패: {e}")
            logger.info("💡 더미 포즈 추정기를 사용합니다")
    
    def estimate_pose(self, image: np.ndarray, frame_id: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """이미지에서 포즈 추정 요청"""
        try:
            # 이미지를 JPEG로 인코딩
            success, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                logger.error("❌ 이미지 인코딩 실패")
                return self._dummy_pose_estimation(image)
            
            # 멀티파트 폼 데이터 준비
            files = {
                'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')
            }
            
            data = {
                'frame_id': str(frame_id),
                'timestamp': str(time.time()),
                'bbox': json.dumps([])  # 전체 이미지 사용
            }
            
            # 포즈 서버에 요청
            response = self.session.post(
                f"{self.pose_server_url}/estimate_pose",
                files=files,
                data=data,
                timeout=3
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 키포인트와 스코어 추출
                keypoints = np.array(result['keypoints'])  # [133, 2]
                scores = np.array(result['scores'])        # [133]
                
                return keypoints, scores
            else:
                logger.warning(f"⚠️ 포즈 서버 응답 오류: HTTP {response.status_code}")
                return self._dummy_pose_estimation(image)
                
        except requests.exceptions.RequestException as e:
            logger.debug(f"포즈 서버 요청 실패: {e}")
            return self._dummy_pose_estimation(image)
        except Exception as e:
            logger.error(f"❌ 포즈 추정 중 오류: {e}")
            return self._dummy_pose_estimation(image)
    
    def _dummy_pose_estimation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """더미 포즈 추정 (폴백)"""
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
    
    def get_stats(self) -> dict:
        """포즈 서버 통계 조회"""
        try:
            response = self.session.get(f"{self.pose_server_url}/stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
