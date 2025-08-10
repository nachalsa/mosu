import cv2
import os
import time
import datetime
from yolo.edge_yolo_detector import EdgeYOLODetector
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

def draw_keypoints_wholebody_on_frame(frame, keypoints, scores, bbox=None, server_image_size=None, threshold=2.0):
    """프레임에 wholebody 키포인트와 스켈레톤을 그리는 함수"""
    num_points = 133
    
    print(f"🔧 좌표 변환 시작")
    print(f"  - 프레임 크기: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  - YOLO bbox: {bbox}")
    print(f"  - 서버 이미지 크기: {server_image_size}")
    print(f"  - 원본 키포인트 타입: {type(keypoints)}, 길이: {len(keypoints)}")
    
    transformed_keypoints = keypoints.copy()
    
    if bbox is not None and server_image_size is not None:
        # YOLO bbox 정보
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = bbox
        yolo_width = yolo_x2 - yolo_x1
        yolo_height = yolo_y2 - yolo_y1
        
        # 서버에서 처리한 이미지 크기 (보통 288x384)
        server_w, server_h = server_image_size
        
        print(f"  - YOLO 크롭 영역: ({yolo_x1}, {yolo_y1}) to ({yolo_x2}, {yolo_y2})")
        print(f"  - YOLO 크롭 크기: {yolo_width}x{yolo_height}")
        print(f"  - 서버 처리 크기: {server_w}x{server_h}")
        
        # 스케일 계산 (서버 이미지 -> YOLO 크롭 영역)
        scale_x = yolo_width / server_w
        scale_y = yolo_height / server_h
        
        print(f"  - 스케일: x={scale_x:.3f}, y={scale_y:.3f}")
        
        transformed_keypoints = []
        for i, kp in enumerate(keypoints):
            if len(kp) >= 2:
                # 서버에서 받은 키포인트 (288x384 기준)
                server_x, server_y = kp[0], kp[1]
                
                # 1단계: 서버 이미지 좌표를 YOLO 크롭 영역 좌표로 변환
                crop_x = server_x * scale_x
                crop_y = server_y * scale_y
                
                # 2단계: YOLO 크롭 영역 좌표를 원본 프레임 좌표로 변환
                orig_x = crop_x + yolo_x1
                orig_y = crop_y + yolo_y1
                
                transformed_keypoints.append([orig_x, orig_y])
                
                # 처음 3개 키포인트의 변환 과정 출력
                if i < 3:
                    print(f"  키포인트 {i}: 서버({server_x:.1f},{server_y:.1f}) -> 크롭({crop_x:.1f},{crop_y:.1f}) -> 원본({orig_x:.1f},{orig_y:.1f})")
            else:
                transformed_keypoints.append(kp)
    else:
        print("  ⚠️ bbox 또는 server_image_size 없음, 키포인트 그대로 사용")

    # 키포인트 그리기
    drawn_points = 0
    for idx in range(min(num_points, len(transformed_keypoints), len(scores))):
        if 17 <= idx <= 22:  # 발 keypoint 무시
            continue
        if scores[idx] > threshold:
            x, y = transformed_keypoints[idx][:2]
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
            x1, y1 = transformed_keypoints[idx1][:2]
            x2, y2 = transformed_keypoints[idx2][:2]
            
            # 프레임 경계 확인
            if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                drawn_lines += 1
    
    print(f"✅ 그려진 연결선: {drawn_lines}개")

class WebcamCapture:
    def __init__(self):
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.capturing_images = False
        self.realtime_translate = False
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
        
        # 스켈레톤 시각화 관련 변수들
        self.show_skeleton = False
        self.last_pose_data = None  # 마지막 포즈 데이터 저장
        self.pose_threshold = 2.0
        
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
    
    def toggle_skeleton_display(self):
        """스켈레톤 표시 토글"""
        self.show_skeleton = not self.show_skeleton
        status = f"스켈레톤 표시: {'켜짐' if self.show_skeleton else '꺼짐'}"
        print(f"🔧 {status}")
        return status
    
    def set_pose_threshold(self, threshold):
        """포즈 임계값 설정"""
        self.pose_threshold = threshold
        return f"포즈 임계값: {threshold}"
    
    def start_realtime(self):
        """이미지 실시간 번역 시작"""
        if not self.realtime_translate:
            self.realtime_translate = True
            self.realtime_fps = 0
            self.last_server_result = ""
            return f"실시간 번역 시작"
        return "이미 실시간 번역 중입니다"

    def stop_realtime(self):
        """이미지 실시간 번역 종료"""
        if self.realtime_translate:
            self.realtime_translate = False
            self.realtime_fps = 0
            return f"실시간 번역 종료"
        return "이미 실시간 번역 중이 아닙니다."

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
            # 실시간 번역 모드에서 YOLO 처리
            if self.realtime_translate:
                self.process_frame_with_yolo(frame)
            
            # 현재 상태 출력 (매 10프레임마다)
            if self.realtime_fps % 10 == 0:
                print(f"📊 현재 상태 - 실시간번역: {self.realtime_translate}, 스켈레톤표시: {self.show_skeleton}, 포즈데이터: {self.last_pose_data is not None}")
            
            # 스켈레톤 표시가 활성화되어 있고 포즈 데이터가 있으면 그리기
            if self.show_skeleton and self.last_pose_data:
                try:
                    keypoints = self.last_pose_data.get('keypoints', [])
                    scores = self.last_pose_data.get('scores', [])
                    bbox = self.last_pose_data.get('bbox', None)
                    server_image_size = self.last_pose_data.get('server_image_size', [288, 384])
                    
                    print(f"🎨 스켈레톤 그리기 시도 - 키포인트: {len(keypoints)}, 스코어: {len(scores)}")
                    
                    if keypoints and scores:
                        draw_keypoints_wholebody_on_frame(
                            frame, keypoints, scores, bbox, server_image_size, self.pose_threshold
                        )
                        print("✅ 스켈레톤 그리기 완료")
                    else:
                        print("❌ 키포인트 또는 스코어가 비어있음")
                        
                except Exception as e:
                    print(f"스켈레톤 그리기 오류: {e}")
            elif self.show_skeleton:
                print("⏳ 스켈레톤 표시 활성화됨, 포즈 데이터 대기 중...")
            
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

            return frame
        return None
    
    def save_image(self, save_skeleton=False):
        """이미지 저장"""
        if not self.initialize_camera():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            # 스켈레톤 저장 옵션이 활성화되어 있고 포즈 데이터가 있으면 그리기
            if save_skeleton and self.last_pose_data:
                try:
                    keypoints = self.last_pose_data.get('keypoints', [])
                    scores = self.last_pose_data.get('scores', [])
                    bbox = self.last_pose_data.get('bbox', None)
                    
                    if keypoints and scores:
                        draw_keypoints_wholebody_on_frame(
                            frame, keypoints, scores, bbox, self.pose_threshold
                        )
                except Exception as e:
                    print(f"스켈레톤 그리기 오류: {e}")
    
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
        """YOLO를 사용한 프레임 처리 및 서버 전송"""
        self.realtime_fps += 1 
        print(f"self.realtime_fps : {self.realtime_fps}")
        s = time.time()
        person_boxes = self.yolo.detect_persons(frame)
        print(f"YOLO 처리 시간: {time.time() - s:.5f}초")
        
        if person_boxes:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes]
            max_idx = int(np.argmax(areas))
            bbox = person_boxes[max_idx]
            crop_img = self.yolo.crop_person_image_rtmw(frame, bbox)
            
            if crop_img is not None:
                ret, buffer = cv2.imencode('.jpg', crop_img)
                if ret:
                    crop_bytes = buffer.tobytes()
                    try:
                        print(f"call server ")

                        files = {'image': (str(self.realtime_fps) + '.jpg', crop_bytes, 'image/jpeg')}
                        data = {'bbox': json.dumps(bbox.tolist() if hasattr(bbox, 'tolist') else bbox)}
                        resp = requests.post(
                            "http://192.168.100.135:5000/estimate_pose",
                            files=files,
                            data=data,
                            timeout=10
                        )
                        
                        # 서버 응답 처리 및 포즈 데이터 저장
                        if resp.status_code == 200:
                            print("서버 전송 성공")
                            try:
                                # 서버 응답 내용 확인
                                print(f"서버 응답 내용: {resp.text[:200]}...")  # 처음 200자만 출력
                                
                                # JSON 파싱 시도
                                response_data = resp.json()
                                print(f"파싱된 응답 키들: {list(response_data.keys())}")
                                
                                # 다양한 키 이름 시도
                                keypoints = None
                                scores = None
                                
                                # 가능한 키 이름들 확인
                                possible_keypoint_keys = ['keypoints', 'original_keypoints', 'poses', 'landmarks']
                                possible_score_keys = ['scores', 'confidences', 'confidence']
                                
                                for key in possible_keypoint_keys:
                                    if key in response_data:
                                        keypoints = response_data[key]
                                        print(f"키포인트 발견: {key}, 길이: {len(keypoints) if keypoints else 0}")
                                        break
                                
                                for key in possible_score_keys:
                                    if key in response_data:
                                        scores = response_data[key]
                                        print(f"스코어 발견: {key}, 길이: {len(scores) if scores else 0}")
                                        break
                                
                                if keypoints and scores:
                                    # 키포인트 데이터 분석
                                    print(f"📊 키포인트 데이터 분석:")
                                    print(f"  - 키포인트 타입: {type(keypoints)}")
                                    print(f"  - 스코어 타입: {type(scores)}")
                                    
                                    # 서버 응답에서 image_size 정보 추출
                                    server_image_size = response_data.get('image_size', [288, 384])
                                    print(f"  - 서버 이미지 크기: {server_image_size}")
                                    
                                    # 키포인트 좌표 범위 확인
                                    if len(keypoints) > 0:
                                        x_coords = [kp[0] for kp in keypoints if len(kp) >= 2]
                                        y_coords = [kp[1] for kp in keypoints if len(kp) >= 2]
                                        if x_coords and y_coords:
                                            print(f"  - X 좌표 범위: {min(x_coords):.1f} ~ {max(x_coords):.1f}")
                                            print(f"  - Y 좌표 범위: {min(y_coords):.1f} ~ {max(y_coords):.1f}")
                                    
                                    # 스코어 범위 확인
                                    if len(scores) > 0:
                                        print(f"  - 스코어 범위: {min(scores):.2f} ~ {max(scores):.2f}")
                                        high_score_count = sum(1 for s in scores if s > self.pose_threshold)
                                        print(f"  - 임계값({self.pose_threshold}) 이상 스코어: {high_score_count}개")
                                    
                                    self.last_pose_data = {
                                        'keypoints': keypoints,
                                        'scores': scores,
                                        'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                                        'server_image_size': server_image_size  # 서버 이미지 크기 추가
                                    }
                                    print(f"✅ 포즈 데이터 업데이트됨! 키포인트: {len(keypoints)}, 스코어: {len(scores)}")
                                else:
                                    print(f"❌ 키포인트 또는 스코어를 찾을 수 없음")
                                    
                            except json.JSONDecodeError as e:
                                print(f"JSON 파싱 오류: {e}")
                                print(f"응답이 JSON이 아님: {resp.text}")
                            except Exception as e:
                                print(f"서버 응답 파싱 오류: {e}")
                        else:
                            print(f"서버 전송 실패: {resp.status_code}")
                            
                        self.last_server_result = str(self.realtime_fps) + "-서버 결과"
                        
                    except Exception as e:
                        print(f"실시간 서버 전송 실패: {e}")
        else:
            print("❌ 사람이 검출되지 않음")

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