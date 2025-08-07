import cv2
import os
import time
import datetime
from yolo.edge_yolo_detector import EdgeYOLODetector
import glob
import requests
import json
import uuid  # 추가된 import
import numpy as np
from typing import List, Tuple

def bbox_xyxy2cs(bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """bbox(x1,y1,x2,y2) -> center, scale 변환"""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    box_w = x2 - x1
    box_h = y2 - y1
    scale = np.array([box_w, box_h], dtype=np.float32)
    return center, scale

def fix_aspect_ratio(scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """scale의 비율을 고정 (width 기준)"""
    w, h = scale
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    return np.array([w, h], dtype=np.float32)

def get_warp_matrix(center, scale, rot, output_size):
    """아핀 변환 행렬 생성"""
    src_w, src_h = scale
    src_dir = get_dir([0, src_h * -0.5], rot)
    dst_w, dst_h = output_size
    dst_dir = np.array([0, dst_h * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    src[2:] = get_third_point(src[0, :], src[1, :])

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2:] = get_third_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(src, dst)
    return trans

def get_dir(src_point, rot_rad):
    """회전된 방향 벡터 계산"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [src_point[0] * cs - src_point[1] * sn,
                  src_point[0] * sn + src_point[1] * cs]
    return np.array(src_result)

def get_third_point(a, b):
    """세 번째 점 계산"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def transform_keypoints_to_original(keypoints: np.ndarray, bbox: List[float]) -> np.ndarray:
    """크롭 좌표(288x384)의 키포인트를 원본 이미지 좌표로 변환"""
    try:
        # RTMW와 동일한 변환 과정을 역변환
        input_width, input_height = 288, 384
        
        # 1. bbox를 center, scale로 변환 (crop 시와 동일)
        bbox_array = np.array(bbox, dtype=np.float32)
        center, scale = bbox_xyxy2cs(bbox_array)
        
        # 2. aspect ratio 고정 (crop 시와 동일)
        aspect_ratio = input_width / input_height  # 0.75
        scale = fix_aspect_ratio(scale, aspect_ratio)
        
        # 3. 아핀 변환 매트릭스 계산 (crop 시와 동일)
        warp_mat = get_warp_matrix(
            center=center,
            scale=scale,
            rot=0.0,
            output_size=(input_width, input_height)
        )
        
        # 4. 역변환 매트릭스 계산
        inv_warp_mat = cv2.invertAffineTransform(warp_mat)
        
        # 5. 키포인트를 homogeneous coordinates로 변환
        num_keypoints = keypoints.shape[0]
        kpts_homo = np.ones((num_keypoints, 3))
        kpts_homo[:, :2] = keypoints[:, :2]
        
        # 6. 역변환 적용
        original_keypoints = np.zeros_like(keypoints)
        for i in range(num_keypoints):
            transformed_pt = inv_warp_mat @ kpts_homo[i]
            original_keypoints[i, 0] = transformed_pt[0]
            original_keypoints[i, 1] = transformed_pt[1]
        
        return original_keypoints
    
    except Exception as e:
        print(f"⚠️ 키포인트 변환 실패: {e}")
        return keypoints  # 실패시 원본 반환

class WebcamCapture:
    def __init__(self):
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.capturing_images = False
        self.capture_folder = None
        self.capture_image_count = 0
        self.video_folder = "./captured_videos"
        self.image_folder = "./captured_images"
        self.video_count = 0
        self.image_count = 0
        self.w = 640
        self.h = 480
        self.fps = 15
        
        self.yolo = EdgeYOLODetector()  # YOLO 인스턴스 추가
        self.crop_folder = None

        # 폴더 생성
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)
        
    def initialize_camera(self):
        """카메라 초기화"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        return self.cap.isOpened()
    
    def release_camera(self):
        """카메라 해제"""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def start_capture_images(self):
        """이미지 연속 저장 시작"""
        if not self.capturing_images:
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.capture_folder = os.path.join("captured_datas", now)
            os.makedirs(self.capture_folder, exist_ok=True)
            self.capturing_images = True
            self.capture_image_count = 0
            return f"이미지 캡처 시작: {self.capture_folder}"
        return "이미 이미지 캡처 중입니다"

    def stop_capture_images(self):
        """이미지 연속 저장 종료"""
        if self.capturing_images:
            self.capturing_images = False
            folder = self.capture_folder
            self.capture_folder = None
            return f"이미지 캡처 종료: {folder}"
        return "이미지 캡처 중이 아닙니다"

    def capture_frame(self):
        """프레임 캡처"""
        if not self.initialize_camera():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            # 녹화 중이면 비디오에 저장
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            # 이미지 연속 저장 중이면 파일로 저장
            if self.capturing_images and self.capture_folder:
                self.capture_image_count += 1
                image_path = os.path.join(
                    self.capture_folder, f"img_{self.capture_image_count:04d}.jpg"
                )
                cv2.imwrite(image_path, frame)

            return frame
        return None
    
    def save_image(self):
        """이미지 저장"""
        frame = self.capture_frame()
        if frame is not None:
            self.image_count += 1
            image_path = f"{self.image_folder}/cam-{self.image_count}.jpg"
            cv2.imwrite(image_path, frame)
            return image_path
        return None
    
    def start_recording(self):
        """비디오 녹화 시작"""
        if not self.recording:
            self.video_count += 1
            video_path = f"{self.video_folder}/record_{self.video_count}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.w, self.h))
            self.recording = True
            return f"녹화 시작: {video_path}"
        return "이미 녹화 중입니다"
    
    def stop_recording(self):
        """비디오 녹화 종료"""
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            return "녹화 종료"
        return "녹화 중이 아닙니다"
    
    def generate_frames(self):
        """스트리밍용 프레임 생성"""
        while True:
            frame = self.capture_frame()
            if frame is not None:
                # JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)  # 약 30fps

    def process_latest_folder_with_yolo(self):
        """가장 최근 폴더의 이미지를 YOLO로 크롭"""
        base_dir = "captured_datas"
        if not os.path.exists(base_dir):
            return "저장된 폴더가 없습니다."
        folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not folders:
            return "저장된 폴더가 없습니다."
        latest_folder = max(folders, key=os.path.getmtime)
        crop_folder = latest_folder + "-crop"
        os.makedirs(crop_folder, exist_ok=True)
        images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")))
        count = 0
        send_count = 0  # 전송 성공 카운트 추가
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            person_boxes = self.yolo.detect_persons(frame)
            for i, bbox in enumerate(person_boxes):
                bbox = person_boxes[0]
                crop_img = self.yolo.crop_person_image_rtmw(frame, bbox)
                if crop_img is not None:
                    crop_path = os.path.join(
                        crop_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}_{i+1}.jpg"
                    )
                    cv2.imwrite(crop_path, crop_img)
                    count += 1

                    # --- 크롭 이미지 서버로 전송 (bbox 포함) ---
                    try:
                        crop_filename = os.path.basename(crop_path)  # crop_filename 정의 추가
                        with open(crop_path, "rb") as f:
                            files = {'image': (crop_filename, f, 'image/jpeg')}
                            data = {
                                'bbox': json.dumps(bbox.tolist() if hasattr(bbox, 'tolist') else bbox),  # bbox를 리스트로 변환
                            }
                            resp = requests.post(
                                "http://192.168.100.135:5000/estimate_pose",
                                files=files,
                                data=data,
                                timeout=10
                            )
                        if resp.status_code == 200:
                            send_count += 1
                            try:
                                response_json = resp.json()
                                keypoints = np.array(response_json.get("keypoints", []), dtype=np.float32)

                                if keypoints.ndim == 2 and keypoints.shape[1] >= 2:
                                    # 원본 이미지 좌표로 복원
                                    original_keypoints = transform_keypoints_to_original(keypoints, bbox)

                                    # 복원된 키포인트를 JSON에 추가
                                    response_json["original_keypoints"] = original_keypoints.tolist()

                                json_filename = os.path.splitext(crop_filename)[0] + f"_pose_{uuid.uuid4().hex[:8]}.json"
                                json_save_path = os.path.join(crop_folder, json_filename)
                                with open(json_save_path, 'w', encoding='utf-8') as f_json:
                                    json.dump(response_json, f_json, indent=2, ensure_ascii=False)
                                print(f"✅ 서버 응답 + 키포인트 변환 저장 완료: {json_save_path}")
                            except Exception as e:
                                print(f"❌ 응답 JSON 파싱 실패: {e}")
                        else:
                            print(f"서버 응답 오류: {resp.status_code} {resp.text}")
                    except Exception as e:
                        print(f"서버 전송 실패: {crop_path} - {e}")
        
        # return 문을 모든 루프 완료 후로 이동하고 send_count 포함
        return f"YOLO 크롭 완료: {count}개 이미지 저장, {send_count}개 서버 전송 성공 ({crop_folder})"