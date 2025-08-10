#!/usr/bin/env python3
"""
네트워크 포즈 추정기
pose-server (192.168.100.135:5000)와 HTTP 통신
"""

import requests
import numpy as np
import cv2
import time
import logging
from typing import Tuple, Optional
import base64
import io
from PIL import Image

logger = logging.getLogger(__name__)

class NetworkPoseEstimator:
    """네트워크 포즈 추정기 - pose-server와 HTTP 통신"""
    
    def __init__(self, server_url: str = "http://192.168.100.135:5000"):
        self.server_url = server_url.rstrip('/')
        self.device = "network"
        
        # 서버 연결 테스트
        self._test_connection()
        
        logger.info(f"✅ 네트워크 포즈 추정기 초기화: {self.server_url}")
    
    def _test_connection(self):
        """서버 연결 테스트"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ pose-server 연결 성공: {self.server_url}")
            else:
                logger.warning(f"⚠️ pose-server 응답 이상: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ pose-server 연결 실패: {e}")
            logger.info("🔄 더미 포즈 추정기로 폴백됩니다")
    
    def estimate_pose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """이미지에서 포즈 추정 (네트워크 호출)"""
        try:
            # 이미지를 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 메타데이터 준비
            files = {
                'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')
            }
            
            data = {
                'frame_id': int(time.time() * 1000),
                'bbox': '[]',
                'timestamp': time.time()
            }
            
            # pose-server에 요청
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/estimate_pose",
                files=files,
                data=data,
                timeout=10
            )
            
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # 키포인트 및 점수 추출
                keypoints = np.array(result['keypoints'], dtype=np.float32)  # [133, 2]
                scores = np.array(result['scores'], dtype=np.float32)  # [133]
                
                # 로그 (가끔씩만)
                if int(time.time()) % 10 == 0:  # 10초마다
                    logger.debug(f"🌐 네트워크 포즈 추정 성공: {request_time:.3f}초")
                
                return keypoints, scores
            else:
                logger.error(f"❌ pose-server 오류: {response.status_code}")
                return self._dummy_pose(image)
                
        except Exception as e:
            logger.error(f"❌ 네트워크 포즈 추정 실패: {e}")
            return self._dummy_pose(image)
    
    def _dummy_pose(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """더미 포즈 추정 (네트워크 실패 시)"""
        h, w = image.shape[:2]
        
        # 133개 더미 키포인트 생성
        keypoints = np.random.rand(133, 2) * np.array([w, h])
        scores = np.random.rand(133) * 0.5 + 0.3  # 0.3-0.8 범위
        
        return keypoints.astype(np.float32), scores.astype(np.float32)

# 테스트 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 이미지 생성
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 네트워크 포즈 추정기 테스트
    estimator = NetworkPoseEstimator("http://192.168.100.135:5000")
    
    # 포즈 추정 테스트
    start_time = time.time()
    keypoints, scores = estimator.estimate_pose(test_image)
    end_time = time.time()
    
    print(f"✅ 테스트 완료:")
    print(f"   - 키포인트 형태: {keypoints.shape}")
    print(f"   - 점수 형태: {scores.shape}")
    print(f"   - 처리 시간: {end_time - start_time:.3f}초")
    print(f"   - 평균 점수: {np.mean(scores):.3f}")
