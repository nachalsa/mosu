import cv2
import os
import time
import datetime
from yolo.edge_yolo_detector import EdgeYOLODetector
import glob
import requests
import json
import numpy as np

class WebcamCapture:
    def __init__(self):
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.capturing_images = False
        self.realtime_translate = False
        self.realtime_fps = 0
        self.capture_folder = None
        self.capture_folder_name = "captured_datas"
        self.capture_image_count = 0
        self.video_folder = "./captured_videos"
        self.image_folder = "./captured_images"
        self.video_count = 0
        self.image_count = 0
        self.w = 640
        self.h = 480
        self.fps = 10
        self.last_server_result = ""
        self.mmpose_server_ip = "192.168.100.135:5000"
        self.rotation_angle = 0  # 0, 90, 180, 270 도 회전
        
        # Set the path to your .hef file
        hailo_model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.hef")  # Adjust filename if different
        
        # Initialize YOLO with both models
        self.yolo = EdgeYOLODetector(
            yolo_model="yolov8n.pt",  # CPU fallback model
            hailo_model="yolov8n.pt" if os.path.exists(hailo_model_path) else None,
            use_hailo=True
        )
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
    
    def start_realtime(self):
        """이미지 실시간 번역 시작"""
        if not self.realtime_translate:
            self.realtime_translate = True
            self.realtime_fps = 0
            self.last_server_result = ""
            return f"실시간 번역 시작"
        return "이미 실시간 번역 중입니다"

    def stop_realtime(self):
        """이미지 실시간 번역 시작"""
        if self.realtime_translate:
            self.realtime_translate = False
            self.realtime_fps = 0
            return f"실시간 번역 종료"
        return "이미 실시간 번역 중이 아닙니다."

    def start_capture_images(self):
        """이미지 연속 저장 시작"""
        if not self.capturing_images:
            now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.capture_folder = os.path.join(self.capture_folder_name, now)
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

    def rotate_image(self, image):
        """이미지 회전"""
        if self.rotation_angle == 0:
            return image
        elif self.rotation_angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation_angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def set_rotation(self, angle):
        """회전 각도 설정"""
        if angle in [0, 90, 180, 270]:
            self.rotation_angle = angle
            return f"화면 회전: {angle}도"
        return "잘못된 회전 각도입니다"
    
    def capture_frame(self):
        """프레임 캡처 (회전 적용)"""
        if not self.initialize_camera():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            # 회전 적용
            frame = self.rotate_image(frame)
            
            # 녹화 중이면 비디오에 저장
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
            # 이미지 연속 저장 중이면 파일로 저장
            elif self.capturing_images and self.capture_folder:
                self.capture_image_count += 1
                image_path = os.path.join(
                    self.capture_folder, f"img_{self.capture_image_count:04d}.jpg"
                )
                cv2.imwrite(image_path, frame)
            elif self.realtime_translate:
                self.realtime_fps += 1 
                crop_img, bbox = self.process_frame_with_yolo(frame)
                if crop_img is not None:
                        ret, buffer = cv2.imencode('.jpg', crop_img)
                        if ret:
                            self.send_pose_server(str(self.realtime_fps) + '.jpg', buffer.tobytes(), bbox)

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
        # fps가 0이거나 None이면 기본값 30fps 사용
        fps = self.fps if self.fps and self.fps > 0 else 30
        target_interval = 1.0 / fps

        while True:
            start = time.time()
            frame = self.capture_frame()
            if frame is not None:
                # JPEG로 인코딩
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            elapsed = time.time() - start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def process_frame_with_yolo(self, frame):
        s = time.time()
        person_boxes = self.yolo.detect_persons(frame)
        print(f"YOLO 처리 시간: {time.time() - s:.5f}초")
        if person_boxes:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes]
            max_idx = int(np.argmax(areas))
            bbox = person_boxes[max_idx]
            ci = self.yolo.crop_person_image_rtmw(frame, bbox)
            return ci, bbox 
        return None, None


    def send_pose_server(self, filename, crop_bytes, bbox):
        # todo 나머지 완료되거나 서버 확정 시 수정해 주세요.
        try:
            files = {'image': (filename, crop_bytes, 'image/jpeg')}
            data = {'bbox': json.dumps(bbox.tolist() if hasattr(bbox, 'tolist') else bbox)}
            resp = requests.post(
                f"http://{self.mmpose_server_ip}/estimate_pose",
                files=files,
                data=data,
                timeout=10
            )
            # 서버 응답을 전역변수 등에 저장 (예시)
            self.last_server_result = str(self.realtime_fps) + "-서버 결과"  # 수정
            if resp.status_code == 200:
                print("서버 전송 성공")
            else:
                print("서버 전송 실패")
        except Exception as e:
            print(f"실시간 서버 전송 실패: {e}")

    def process_latest_folder_with_yolo(self):
        """가장 최근 폴더의 이미지를 YOLO로 크롭"""
        #가장 최근 폴더 가져오기
        base_dir = self.capture_folder_name
        folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not folders:
            return "저장된 폴더가 없습니다."
        latest_folder = max(folders, key=os.path.getmtime)
        #디버그 크롭용 폴더 생성
        crop_folder = self.create_crop_folder(latest_folder)

        images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")))
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            crop_img, bbox = self.process_frame_with_yolo(frame)

            if crop_img is not None:
                ret, buffer = cv2.imencode('.jpg', crop_img)
                if ret:
                    self.send_pose_server(str(self.realtime_fps) + '.jpg', buffer.tobytes(), bbox)
                    # 디버그용 크롭 이미지 파일 저장
                    crop_filename = os.path.basename(img_path).replace('.jpg', '_crop.jpg')
                    crop_path = os.path.join(crop_folder, crop_filename)
                    cv2.imwrite(crop_path, crop_img)
                    
        return f"YOLO 크롭 완료"

    def create_crop_folder(self, latest_folder):
        # crop_folder 하위에 latest_folder 이름으로 폴더 생성
        latest_folder_name = os.path.basename(latest_folder)
        crop_folder = os.path.join("captured_cropped", latest_folder_name)
        os.makedirs(crop_folder, exist_ok=True)
        return crop_folder