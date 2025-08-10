import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from collections import deque

# Add Hailo imports
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                                InferVStreams, ConfigureParams)
    HAILO_AVAILABLE = True
    print("✅ Hailo platform available")
except ImportError:
    print("⚠️ Hailo platform not available, falling back to CPU")
    HAILO_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ ultralytics 미설치 - pip install ultralytics")
    YOLO_AVAILABLE = False

class EdgeYOLODetector:
    """엣지 디바이스용 YOLO 검출기 (Hailo-8 가속 지원)"""
    
    def __init__(self, 
                 yolo_model: str = "yolov8n.pt",
                 hailo_model: str = "yolov8n.hef",  # .hef 파일 경로
                 use_hailo: bool = True,
                 conf_thresh: float = 0.6,
                 iou_thresh: float = 0.6,
                 max_det: int = 1,
                 img_size: int = 320):
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.img_size = img_size
        self.use_hailo = use_hailo and HAILO_AVAILABLE
        
        if self.use_hailo and hailo_model:
            self._init_hailo(hailo_model)
        else:
            self._init_ultralytics(yolo_model)
    
    def _init_hailo(self, hailo_model: str):
        """Hailo 모델 초기화"""
        try:
            print(f"🔧 Hailo 모델 로딩 중: {hailo_model}")
            
            # HEF 파일 로드
            self.hef = HEF(hailo_model)
            
            # VDevice 생성
            self.target = VDevice()
            
            # 네트워크 그룹 설정
            self.network_group = self.target.configure(self.hef)[0]
            self.network_group_params = self.network_group.create_params()
            
            # 입력/출력 스트림 설정
            self.input_vstreams_params = self.network_group.make_input_vstream_params(
                quantized=False, format_type=HailoStreamInterface.UINT8
            )
            self.output_vstreams_params = self.network_group.make_output_vstream_params(
                quantized=False, format_type=HailoStreamInterface.FLOAT32
            )
            
            print("✅ Hailo 모델 로딩 완료")
            self.backend = "hailo"
            
        except Exception as e:
            print(f"❌ Hailo 초기화 실패: {e}")
            print("CPU 모드로 전환합니다...")
            self._init_ultralytics("yolov8n.pt")
    
    def _init_ultralytics(self, yolo_model: str):
        """Ultralytics 모델 초기화 (CPU/GPU 백엔드)"""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics가 필요합니다: pip install ultralytics")
        
        print(f"🔧 {yolo_model} 모델 로딩 중...")
        self.model = YOLO(yolo_model)
        print(f"✅ {yolo_model} 모델 로딩 완료")
        self.backend = "ultralytics"
    
    def detect_persons(self, image: np.ndarray) -> List[List[float]]:
        """사람 검출 (Hailo 또는 Ultralytics 사용)"""
        if self.backend == "hailo":
            return self._detect_persons_hailo(image)
        else:
            return self._detect_persons_ultralytics(image)
    
    def _detect_persons_hailo(self, image: np.ndarray) -> List[List[float]]:
        """Hailo를 사용한 사람 검출"""
        try:
            # 이미지 전처리
            input_data = self._preprocess_for_hailo(image)
            
            # Hailo 추론
            with InferVStreams(self.network_group, 
                             self.input_vstreams_params, 
                             self.output_vstreams_params) as infer_pipeline:
                
                # 추론 실행
                outputs = infer_pipeline.infer({
                    list(self.input_vstreams_params.keys())[0]: input_data
                })
                
                # 후처리
                person_boxes = self._postprocess_hailo_output(outputs, image.shape)
                return person_boxes
                
        except Exception as e:
            print(f"❌ Hailo 검출 실패: {e}")
            return []
    
    def _detect_persons_ultralytics(self, image: np.ndarray) -> List[List[float]]:
        """기존 Ultralytics 검출 방식"""
        try:
            results = self.model(
                image,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=1,
                classes=[0],
                verbose=False,
                imgsz=self.img_size
            )
            
            person_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    person_coords = boxes.xyxy
                    
                    if hasattr(person_coords, 'cpu'):
                        person_coords = person_coords.cpu().numpy()
                    
                    person_boxes.extend(person_coords.tolist())
            
            return person_boxes
        
        except Exception as e:
            print(f"❌ YOLO 검출 실패: {e}")
            return []
    
    def _preprocess_for_hailo(self, image: np.ndarray) -> np.ndarray:
        """Hailo 입력을 위한 전처리"""
        # 이미지 크기 조정 (Hailo 모델의 입력 크기에 맞춤)
        resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # BGR to RGB 변환
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 -> 0-1)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # 배치 차원 추가 [H,W,C] -> [1,H,W,C]
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
    
    def _postprocess_hailo_output(self, outputs: dict, original_shape: tuple) -> List[List[float]]:
        """Hailo 출력 후처리"""
        # 이 부분은 사용하는 YOLO 모델의 출력 형식에 따라 수정 필요
        # 일반적으로 [batch, boxes, 5+classes] 형태
        
        person_boxes = []
        
        # 출력에서 첫 번째 텐서 가져오기 (보통 detection 결과)
        detection_output = list(outputs.values())[0]
        
        # 후처리 로직 (NMS, 좌표 변환 등)
        # 이 부분은 실제 Hailo 모델의 출력 형식에 맞게 구현해야 함
        
        return person_boxes
    
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