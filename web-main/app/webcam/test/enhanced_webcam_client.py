import cv2
import os
import time
import datetime
import glob
import requests
import json
import numpy as np

# COCO Wholebody 스켈레톤 연결 정보
COCO_WHOLEBODY_SKELETON = [
    # Body (0~16)
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],

    # Left Hand (91~111)
    [91, 92], [92, 93], [93, 94], [94, 95],         # Thumb
    [91, 96], [96, 97], [97, 98], [98, 99],         # Index
    [91, 100], [100, 101], [101, 102], [102, 103],  # Middle
    [91, 104], [104, 105], [105, 106], [106, 107],  # Ring
    [91, 108], [108, 109], [109, 110], [110, 111],  # Pinky

    # Right Hand (112~132)
    [112, 113], [113, 114], [114, 115], [115, 116],      # Thumb
    [112, 117], [117, 118], [118, 119], [119, 120],      # Index
    [112, 121], [121, 122], [122, 123], [123, 124],      # Middle
    [112, 125], [125, 126], [126, 127], [127, 128],      # Ring
    [112, 129], [129, 130], [130, 131], [131, 132],      # Pinky
]

def draw_keypoints_wholebody_on_frame(frame, keypoints, scores, threshold=2.0):
    """프레임에 wholebody 키포인트와 스켈레톤을 그리는 함수 (서버에서 이미 좌표 변환됨)"""
    num_points = 133
    
    print(f"🔧 스켈레톤 그리기 시작")
    print(f"  - 프레임 크기: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  - 키포인트 개수: {len(keypoints)}")
    print(f"  - 스코어 개수: {len(scores)}")

    # 키포인트 그리기 (서버에서 이미 원본 이미지 좌표로 변환됨)
    drawn_points = 0
    for idx in range(min(num_points, len(keypoints), len(scores))):
        if 17 <= idx <= 22:  # 발 keypoint 무시
            continue
        if scores[idx] > threshold:
            x, y = keypoints[idx][:2]
            # 프레임 경계 확인
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                drawn_points += 1

    print(f"✅ 그려진 키포인트: {drawn_points}개")

    # 스켈레톤 연결선 그리기
    drawn_lines = 0
    for idx1, idx2 in COCO_WHOLEBODY_SKELETON:
        if 17 <= idx1 <= 22 or 17 <= idx2 <= 22:  # 발 keypoint 포함된 연결 무시
            continue
        if (idx1 < len(scores) and idx2 < len(scores) and 
            scores[idx1] > threshold and scores[idx2] > threshold):
            x1, y1 = keypoints[idx1][:2]
            x2, y2 = keypoints[idx2][:2]
            
            # 프레임 경계 확인
            if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                drawn_lines += 1
    
    print(f"✅ 그려진 연결선: {drawn_lines}개")

def draw_mediapipe_landmarks(frame, hand_landmarks, pose_landmarks):
    """MediaPipe 랜드마크를 프레임에 그리는 함수"""
    h, w = frame.shape[:2]
    
    # 손 랜드마크 그리기
    if hand_landmarks:
        for hand_data in hand_landmarks:
            hand_color = (0, 255, 255) if hand_data['handedness'] == 'Left' else (255, 255, 0)
            landmarks = hand_data['landmarks']
            
            # 손가락 연결선 그리기
            hand_connections = [
                # Thumb
                [0, 1], [1, 2], [2, 3], [3, 4],
                # Index
                [0, 5], [5, 6], [6, 7], [7, 8],
                # Middle  
                [0, 9], [9, 10], [10, 11], [11, 12],
                # Ring
                [0, 13], [13, 14], [14, 15], [15, 16],
                # Pinky
                [0, 17], [17, 18], [18, 19], [19, 20],
            ]
            
            # 연결선 그리기
            for connection in hand_connections:
                if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                    x1 = int(landmarks[connection[0]]['x'] * w)
                    y1 = int(landmarks[connection[0]]['y'] * h)
                    x2 = int(landmarks[connection[1]]['x'] * w)
                    y2 = int(landmarks[connection[1]]['y'] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), hand_color, 2)
            
            # 키포인트 그리기
            for landmark in landmarks:
                x = int(landmark['x'] * w)
                y = int(landmark['y'] * h)
                cv2.circle(frame, (x, y), 3, hand_color, -1)
    
    # 포즈 랜드마크 그리기 (상체만)
    if pose_landmarks:
        pose_color = (255, 0, 255)  # 마젠타
        
        # 상체 연결선
        pose_connections = [
            [11, 12],  # 어깨
            [11, 13], [13, 15],  # 왼쪽 팔
            [12, 14], [14, 16],  # 오른쪽 팔
            [11, 23], [12, 24],  # 몸통
            [23, 24]  # 엉덩이
        ]
        
        # 인덱스를 실제 데이터 위치로 매핑
        pose_dict = {landmark['index']: landmark for landmark in pose_landmarks}
        
        # 연결선 그리기
        for connection in pose_connections:
            if connection[0] in pose_dict and connection[1] in pose_dict:
                landmark1 = pose_dict[connection[0]]
                landmark2 = pose_dict[connection[1]]
                
                x1 = int(landmark1['x'] * w)
                y1 = int(landmark1['y'] * h)
                x2 = int(landmark2['x'] * w)
                y2 = int(landmark2['y'] * h)
                
                if landmark1['visibility'] > 0.5 and landmark2['visibility'] > 0.5:
                    cv2.line(frame, (x1, y1), (x2, y2), pose_color, 2)
        
        # 키포인트 그리기
        for landmark in pose_landmarks:
            if landmark['visibility'] > 0.5:
                x = int(landmark['x'] * w)
                y = int(landmark['y'] * h)
                cv2.circle(frame, (x, y), 4, pose_color, -1)

class EnhancedWebcamCapture:
    def __init__(self, server_host="192.168.100.135", server_port=5000):
        self.server_host = server_host
        self.server_port = server_port
        self.server_url_pose = f"http://{server_host}:{server_port}/estimate_pose"
        self.server_url_sign = f"http://{server_host}:{server_port}/recognize_sign"
        self.server_url_hybrid = f"http://{server_host}:{server_port}/hybrid_analysis"
        
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.capturing_images = False
        
        # 실시간 처리 모드
        self.realtime_pose = False       # RTMW 포즈 추정
        self.realtime_sign = False       # MediaPipe 수어 인식
        self.realtime_hybrid = False     # 하이브리드 분석
        
        self.realtime_fps = 0
        self.capture_folder = None
        self.capture_image_count = 0
        self.video_folder = "./captured_videos"
        self.image_folder = "./captured_images"
        self.video_count = 0
        self.image_count = 0
        self.w = 640
        self.h = 480
        self.fps = 10
        self.last_server_result = ""
        
        # 시각화 관련 변수들
        self.show_skeleton = False      # RTMW 스켈레톤 표시
        self.show_mediapipe = False     # MediaPipe 랜드마크 표시
        self.last_pose_data = None      # 마지막 RTMW 포즈 데이터
        self.last_sign_data = None      # 마지막 MediaPipe 데이터
        self.last_prediction = ""       # 마지막 수어 예측 결과
        self.pose_threshold = 2.0

        # 폴더 생성
        os.makedirs(self.video_folder, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)

        print(f"✅ Enhanced 웹캠 캡처 초기화 완료")
        print(f"   - 서버 주소: {server_host}:{server_port}")
        print(f"   - 지원 기능: RTMW 포즈 추정, MediaPipe 수어 인식, 하이브리드 분석")

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
    
    def toggle_skeleton_display(self):
        """RTMW 스켈레톤 표시 토글"""
        self.show_skeleton = not self.show_skeleton
        status = f"RTMW 스켈레톤 표시: {'켜짐' if self.show_skeleton else '꺼짐'}"
        print(f"🔧 {status}")
        return status
    
    def toggle_mediapipe_display(self):
        """MediaPipe 랜드마크 표시 토글"""
        self.show_mediapipe = not self.show_mediapipe
        status = f"MediaPipe 랜드마크 표시: {'켜짐' if self.show_mediapipe else '꺼짐'}"
        print(f"🔧 {status}")
        return status
    
    def set_pose_threshold(self, threshold):
        """포즈 임계값 설정"""
        self.pose_threshold = threshold
        return f"포즈 임계값: {threshold}"
    
    # 실시간 처리 모드 제어
    def start_realtime_pose(self):
        """RTMW 실시간 포즈 추정 시작"""
        if not self.realtime_pose:
            self.realtime_pose = True
            self.realtime