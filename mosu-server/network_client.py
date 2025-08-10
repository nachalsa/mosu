#!/usr/bin/env python3
"""
네트워크 통합 클라이언트 - 웹서버에서 pose-server와 mosu-server와 통신
"""

import cv2
import numpy as np
import requests
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import base64
import io
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)

class NetworkSignLanguageClient:
    """통합 수화 인식 네트워크 클라이언트"""
    
    def __init__(self, 
                 pose_server_url: str = "http://192.168.100.135:5000",
                 mosu_server_url: str = "http://192.168.100.26:8002"):
        
        self.pose_server_url = pose_server_url.rstrip('/')
        self.mosu_server_url = mosu_server_url.rstrip('/')
        
        # HTTP 세션 생성
        self.session = requests.Session()
        self.session.timeout = 10
        
        # 서버 연결 테스트
        self.pose_server_available = self._test_server_connection(self.pose_server_url, "포즈 서버")
        self.mosu_server_available = self._test_server_connection(self.mosu_server_url, "MOSU 서버")
        
        logger.info(f"✅ 네트워크 클라이언트 초기화 완료")
        logger.info(f"   - 포즈 서버: {'✅' if self.pose_server_available else '❌'} {self.pose_server_url}")
        logger.info(f"   - MOSU 서버: {'✅' if self.mosu_server_available else '❌'} {self.mosu_server_url}")
    
    def _test_server_connection(self, url: str, name: str) -> bool:
        """서버 연결 테스트"""
        try:
            response = self.session.get(f"{url}/health", timeout=3)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ {name} 연결 성공 - 상태: {data.get('status', 'unknown')}")
                return True
            else:
                logger.warning(f"⚠️ {name} 응답 오류: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ {name} 연결 실패: {e}")
            return False
    
    def process_frame_with_pose_server(self, image: np.ndarray, frame_id: int = 0) -> Optional[Dict[str, Any]]:
        """포즈 서버를 통한 프레임 처리"""
        if not self.pose_server_available:
            return None
        
        try:
            # 이미지를 JPEG로 인코딩
            success, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("❌ 이미지 인코딩 실패")
                return None
            
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
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'keypoints': result.get('keypoints', []),
                    'scores': result.get('scores', []),
                    'processing_time': result.get('processing_time', 0),
                    'device': result.get('device', 'unknown')
                }
            else:
                logger.warning(f"⚠️ 포즈 서버 응답 오류: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.debug(f"포즈 서버 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 포즈 추정 중 오류: {e}")
            return None
    
    def process_frame_with_mosu_server(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """MOSU 서버를 통한 통합 처리 (포즈 + 수화 인식)"""
        if not self.mosu_server_available:
            return None
        
        try:
            # 이미지를 Base64로 인코딩
            success, img_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                logger.error("❌ 이미지 인코딩 실패")
                return None
            
            # Base64 인코딩
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{img_base64}"
            
            # MOSU 서버에 WebSocket 대신 REST API 요청 (임시)
            # 실제로는 WebSocket을 사용해야 하지만, 여기서는 HTTP로 시뮬레이션
            
            # 임시로 포즈만 처리하고 결과 반환
            pose_result = self.process_frame_with_pose_server(image)
            
            if pose_result:
                return {
                    'pose': pose_result,
                    'sign': {
                        'word': None,  # 실제로는 MOSU 서버에서 수화 인식 결과
                        'confidence': 0.0
                    },
                    'source': 'network_integrated'
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ MOSU 서버 처리 중 오류: {e}")
            return None
    
    def get_pose_server_stats(self) -> Dict[str, Any]:
        """포즈 서버 통계 조회"""
        if not self.pose_server_available:
            return {"error": "포즈 서버 연결 안됨"}
        
        try:
            response = self.session.get(f"{self.pose_server_url}/stats", timeout=3)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_mosu_server_stats(self) -> Dict[str, Any]:
        """MOSU 서버 통계 조회"""
        if not self.mosu_server_available:
            return {"error": "MOSU 서버 연결 안됨"}
        
        try:
            response = self.session.get(f"{self.mosu_server_url}/stats", timeout=3)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_server_status(self) -> Dict[str, Any]:
        """전체 서버 상태 조회"""
        return {
            "pose_server": {
                "available": self.pose_server_available,
                "url": self.pose_server_url
            },
            "mosu_server": {
                "available": self.mosu_server_available,
                "url": self.mosu_server_url
            },
            "network_info": {
                "web_server": "192.168.100.90",
                "pose_server": "192.168.100.135:5000", 
                "mosu_server": "192.168.100.26:8002"
            }
        }

# 사용 예시 및 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 네트워크 클라이언트 생성
    client = NetworkSignLanguageClient()
    
    # 서버 상태 확인
    status = client.get_server_status()
    print("🔍 서버 상태:")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    # 더미 이미지로 테스트
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n🧪 포즈 서버 테스트...")
    pose_result = client.process_frame_with_pose_server(test_image)
    if pose_result:
        print(f"✅ 포즈 추정 성공: {len(pose_result.get('keypoints', []))}개 키포인트")
    else:
        print("❌ 포즈 추정 실패")
    
    print("\n🧪 MOSU 서버 테스트...")
    mosu_result = client.process_frame_with_mosu_server(test_image)
    if mosu_result:
        print(f"✅ 통합 처리 성공")
    else:
        print("❌ 통합 처리 실패")
