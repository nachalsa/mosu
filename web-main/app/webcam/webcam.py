import cv2
import os
import time

class WebcamCapture:
    def __init__(self):
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.video_folder = "./captured_videos"
        self.image_folder = "./captured_images"
        self.video_count = 0
        self.image_count = 0
        self.w = 640
        self.h = 480
        self.fps = 30
        
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
    
    def capture_frame(self):
        """프레임 캡처"""
        if not self.initialize_camera():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            # 녹화 중이면 비디오에 저장
            if self.recording and self.video_writer:
                self.video_writer.write(frame)
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
