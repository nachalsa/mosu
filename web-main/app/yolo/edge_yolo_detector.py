
import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from collections import deque

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

class EdgeYOLODetector:
    """엣지 디바이스용 YOLO 검출기"""
    
    def __init__(self, 
                 yolo_model: str = "yolo11l.pt",
                 conf_thresh: float = 0.4,
                 iou_thresh: float = 0.6,
                 max_det: int = 50,
                 img_size: int = 832):
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.img_size = img_size
        
        # YOLO 모델 로드
        print(f"🔧 YOLO11L 모델 로딩 중...")
        self.model = YOLO(yolo_model)
        print(f"✅ YOLO11L 모델 로딩 완료")
    
    def detect_persons(self, image: np.ndarray) -> List[List[float]]:
        """사람 검출"""
        try:
            results = self.model(
                image,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                classes=[0],  # 사람 클래스만
                verbose=False,
                imgsz=self.img_size
            )
            
            person_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy
                    person_confs = boxes.conf
                    
                    # 신뢰도 재필터링
                    conf_mask = person_confs >= self.conf_thresh
                    if conf_mask.any():
                        filtered_boxes = person_coords[conf_mask]
                        filtered_confs = person_confs[conf_mask]
                        
                        # numpy 변환
                        if hasattr(filtered_boxes, 'cpu'):
                            filtered_boxes = filtered_boxes.cpu().numpy()
                            filtered_confs = filtered_confs.cpu().numpy()
                        
                        # 신뢰도순 정렬
                        sorted_indices = np.argsort(filtered_confs)[::-1]
                        sorted_boxes = filtered_boxes[sorted_indices]
                        
                        person_boxes.extend(sorted_boxes.tolist())
            
            return person_boxes
            
        except Exception as e:
            print(f"❌ YOLO 검출 실패: {e}")
            return []
    
    def crop_person_image_rtmw(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """RTMW 방식으로 사람 이미지 크롭"""
        try:
            # RTMW 설정: width=288, height=384
            input_width, input_height = 288, 384
            
            # 1. bbox를 center, scale로 변환
            bbox_array = np.array(bbox, dtype=np.float32)
            center, scale = self.bbox_xyxy2cs(bbox_array)
            
            # 2. aspect ratio 고정
            aspect_ratio = input_width / input_height  # 0.75
            scale = self.fix_aspect_ratio(scale, aspect_ratio)
            
            # 3. 아핀 변환 매트릭스 계산
            warp_mat = self.get_warp_matrix(
                center=center,
                scale=scale,
                rot=0.0,
                output_size=(input_width, input_height)
            )
            
            # 4. 아핀 변환 적용
            cropped_image = cv2.warpAffine(
                image, 
                warp_mat, 
                (input_width, input_height), 
                flags=cv2.INTER_LINEAR
            )
            
            return cropped_image
                
        except Exception as e:
            print(f"⚠️ 크롭 실패: {e}")
            return None

    # RTMW 전처리 함수들 (streamlined_processor.py에서 가져옴)
    def bbox_xyxy2cs(self, bbox: np.ndarray, padding: float = 1.10) -> Tuple[np.ndarray, np.ndarray]:
        """바운딩박스를 center, scale로 변환"""
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]
        
        scale = (bbox[..., 2:] - bbox[..., :2]) * padding
        center = (bbox[..., 2:] + bbox[..., :2]) * 0.5
        
        if dim == 1:
            center = center[0]
            scale = scale[0]
        
        return center, scale

    def _rotate_point(self, pt: np.ndarray, angle_rad: float) -> np.ndarray:
        """점을 회전"""
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        return np.array([pt[0] * cos_val - pt[1] * sin_val,
                        pt[0] * sin_val + pt[1] * cos_val])

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """세 번째 점을 계산 (직교점)"""
        direction = a - b
        return b + np.array([-direction[1], direction[0]])

    def get_warp_matrix(self, center: np.ndarray, scale: np.ndarray, rot: float, 
                    output_size: Tuple[int, int]) -> np.ndarray:
        """아핀 변환 매트릭스 계산"""
        src_w, src_h = scale[:2]
        dst_w, dst_h = output_size[:2]
        
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
        dst_dir = np.array([dst_w * -0.5, 0.])
        
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])
        
        warp_mat = cv2.getAffineTransform(src, dst)
        return warp_mat

    def fix_aspect_ratio(self, bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
        """bbox를 고정 종횡비로 조정"""
        w, h = bbox_scale[0], bbox_scale[1]
        if w > h * aspect_ratio:
            new_h = w / aspect_ratio
            bbox_scale = np.array([w, new_h])
        else:
            new_w = h * aspect_ratio
            bbox_scale = np.array([new_w, h])
        return bbox_scale