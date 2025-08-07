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

        # crop_folder 하위에 latest_folder 이름으로 폴더 생성
        latest_folder_name = os.path.basename(latest_folder)
        crop_folder = os.path.join("captured_cropped", latest_folder_name)
        os.makedirs(crop_folder, exist_ok=True)
        
        images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")))
        count = 0
        send_count = 0  # 전송 성공 카운트 추가
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            st = time.time()
            person_boxes = self.yolo.detect_persons(frame)
            et = time.time()
            print(f"YOLO 처리 시간: {et - st:.5f}초, 이미지: {img_path}")
            if person_boxes:
                # 각 bbox: [x1, y1, x2, y2]
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes]
                max_idx = int(np.argmax(areas))
                bbox = person_boxes[max_idx]
                crop_img = self.yolo.crop_person_image_rtmw(frame, bbox)
                if crop_img is not None:
                    # 디버그용 크롭 이미지 파일 저장
                    crop_filename = os.path.basename(img_path).replace('.jpg', '_crop.jpg')
                    crop_path = os.path.join(crop_folder, crop_filename)
                    cv2.imwrite(crop_path, crop_img)
                    
                    # 파일로 저장하지 않고 메모리에서 JPEG 인코딩
                    ret, buffer = cv2.imencode('.jpg', crop_img)
                    if not ret:
                        print(f"이미지 인코딩 실패: {img_path}")
                        continue
                    crop_bytes = buffer.tobytes()

                    # --- 크롭 이미지 서버로 전송 (bbox 포함) ---
                    try:
                        files = {'image': (os.path.basename(img_path), crop_bytes, 'image/jpeg')}
                        data = {
                            'bbox': json.dumps(bbox.tolist() if hasattr(bbox, 'tolist') else bbox),
                        }
                        resp = requests.post(
                            "http://192.168.100.135:5000/estimate_pose",
                            files=files,
                            data=data,
                            timeout=10
                        )
                        if resp.status_code == 200:
                            send_count += 1
                            print(f"서버로 전송 성공: {send_count} , {img_path} ")
                        else:
                            print(f"서버 응답 오류: {resp.status_code} {resp.text}")
                    except Exception as e:
                        print(f"서버 전송 실패: {crop_filename} - {e}")
        
        # return 문을 모든 루프 완료 후로 이동하고 send_count 포함
        return f"YOLO 크롭 완료: {count}개 이미지 저장, {send_count}개 서버 전송 성공"