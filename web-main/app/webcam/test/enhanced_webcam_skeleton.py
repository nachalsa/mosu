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
    """프레임에 wholebody 키포인트와 스켈레톤을 그리는 함수 (서버에서 이미  좌표 변환됨)"""
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

def draw_sign_recognition_result(frame, sign_result, confidence_threshold=0.6):
    """프레임에 수어 인식 결과를 그리는 함수"""
    if not sign_result:
        return
    
    try:
        h, w = frame.shape[:2]
        
        # 반투명 배경 박스 그리기
        overlay = frame.copy()
        box_height = 120
        cv2.rectangle(overlay, (10, 10), (w-10, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목 텍스트
        cv2.putText(frame, "Sign Recognition Result", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 예측 결과가 있는 경우
        sign_prediction = sign_result.get('sign_prediction')
        if sign_prediction and sign_prediction.get('confidence', 0) >= confidence_threshold:
            predicted_class = sign_prediction.get('predicted_class', 'Unknown')
            confidence = sign_prediction.get('confidence', 0)
            
            # 예측된 수어와 신뢰도 표시
            result_text = f"Sign: {predicted_class}"
            confidence_text = f"Confidence: {confidence:.2f}"
            
            cv2.putText(frame, result_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # 예측 결과가 없거나 신뢰도가 낮은 경우
            cv2.putText(frame, "No sign detected or low confidence", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # MediaPipe 감지 상태 표시
        mediapipe_results = sign_result.get('mediapipe_results', {})
        hand_detected = mediapipe_results.get('hand_detected', False)
        pose_detected = mediapipe_results.get('pose_detected', False)
        
        status_text = f"Hand: {'✓' if hand_detected else '✗'}  Pose: {'✓' if pose_detected else '✗'}"
        cv2.putText(frame, status_text, (w-200, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    except Exception as e:
        print(f"❌ 수어 인식 결과 표시 오류: {e}")

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

        # 수어 인식 관련 변수들 추가
        self.show_sign_recognition = False
        self.sign_recognition_mode = False  # 수어 인식 모드 활성화
        self.last_sign_result = None  # 마지막 수어 인식 결과
        self.sign_confidence_threshold = 0.6  # 수어 인식 신뢰도 임계값
        self.use_mediapipe_server = True  # MediaPipe 서버 사용 여부

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
    
    def toggle_sign_recognition_display(self):
        """수어 인식 결과 표시 토글"""
        self.show_sign_recognition = not self.show_sign_recognition
        status = f"수어 인식 결과 표시: {'켜짐' if self.show_sign_recognition else '꺼짐'}"
        print(f"🔧 {status}")
        return status
    
    def toggle_sign_recognition_mode(self):
        """수어 인식 모드 토글"""
        self.sign_recognition_mode = not self.sign_recognition_mode
        if self.sign_recognition_mode:
            self.show_sign_recognition = True  # 자동으로 결과 표시도 활성화
        status = f"수어 인식 모드: {'켜짐' if self.sign_recognition_mode else '꺼짐'}"
        print(f"🔧 {status}")
        return status
    
    def set_pose_threshold(self, threshold):
        """포즈 임계값 설정"""
        self.pose_threshold = threshold
        return f"포즈 임계값: {threshold}"
    
    def set_sign_confidence_threshold(self, threshold):
        """수어 인식 신뢰도 임계값 설정"""
        self.sign_confidence_threshold = threshold
        return f"수어 인식 신뢰도 임계값: {threshold}"
    
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
            # 실시간 번역 모드에서 서버로 원본 이미지 전송
            if self.realtime_translate:
                self.send_frame_to_server(frame)
            
            # 수어 인식 모드가 활성화된 경우 MediaPipe 서버로 전송
            if self.sign_recognition_mode:
                self.send_frame_for_sign_recognition(frame)
            
            # 현재 상태 출력 (매 10프레임마다)
            if self.realtime_fps % 10 == 0:
                print(f"📊 현재 상태 - 실시간번역: {self.realtime_translate}, 수어인식: {self.sign_recognition_mode}, 스켈레톤표시: {self.show_skeleton}, 수어결과표시: {self.show_sign_recognition}")
            
            # 스켈레톤 표시가 활성화되어 있고 포즈 데이터가 있으면 그리기
            if self.show_skeleton and self.last_pose_data:
                try:
                    keypoints = self.last_pose_data.get('keypoints', [])
                    scores = self.last_pose_data.get('scores', [])
                    
                    print(f"🎨 스켈레톤 그리기 시도 - 키포인트: {len(keypoints)}, 스코어: {len(scores)}")
                    
                    if keypoints and scores:
                        draw_keypoints_wholebody_on_frame(
                            frame, keypoints, scores, self.pose_threshold
                        )
                        print("✅ 스켈레톤 그리기 완료")
                    else:
                        print("❌ 키포인트 또는 스코어가 비어있음")
                        
                except Exception as e:
                    print(f"스켈레톤 그리기 오류: {e}")
            elif self.show_skeleton:
                print("⏳ 스켈레톤 표시 활성화됨, 포즈 데이터 대기 중...")
            
            # 수어 인식 결과 표시
            if self.show_sign_recognition and self.last_sign_result:
                try:
                    draw_sign_recognition_result(
                        frame, self.last_sign_result, self.sign_confidence_threshold
                    )
                    print("✅ 수어 인식 결과 표시 완료")
                except Exception as e:
                    print(f"수어 인식 결과 표시 오류: {e}")
            
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
    
    def save_image(self, save_skeleton=False, save_sign_result=False):
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
                    
                    if keypoints and scores:
                        draw_keypoints_wholebody_on_frame(
                            frame, keypoints, scores, self.pose_threshold
                        )
                except Exception as e:
                    print(f"스켈레톤 그리기 오류: {e}")
            
            # 수어 인식 결과 저장 옵션
            if save_sign_result and self.last_sign_result:
                try:
                    draw_sign_recognition_result(
                        frame, self.last_sign_result, self.sign_confidence_threshold
                    )
                except Exception as e:
                    print(f"수어 인식 결과 그리기 오류: {e}")
    
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

    def send_frame_for_sign_recognition(self, frame):
        """MediaPipe 수어 인식을 위해 프레임을 서버로 전송"""
        try:
            # 원본 이미지를 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("❌ 이미지 인코딩 실패 (수어 인식)")
                return
                
            frame_bytes = buffer.tobytes()
            
            print(f"📤 수어 인식을 위해 서버로 이미지 전송")
            
            files = {'image': (f'sign_{self.realtime_fps}.jpg', frame_bytes, 'image/jpeg')}
            data = {
                'frame_id': str(self.realtime_fps),
                'confidence_threshold': str(self.sign_confidence_threshold),
                'include_features': 'false'  # 특징 데이터는 포함하지 않음
            }
            
            resp = requests.post(
                "http://192.168.100.135:5000/recognize_sign",  # 새로운 수어 인식 엔드포인트
                files=files,
                data=data,
                timeout=10
            )
            
            # 서버 응답 처리 및 수어 인식 결과 저장
            if resp.status_code == 200:
                print("✅ 수어 인식 서버 전송 성공")
                try:
                    # 서버 응답 내용 확인
                    print(f"📥 수어 인식 서버 응답 내용: {resp.text[:300]}...")
                    
                    # JSON 파싱
                    response_data = resp.json()
                    print(f"📋 수어 인식 응답 키들: {list(response_data.keys())}")
                    
                    # 오류 체크
                    if 'error' in response_data:
                        print(f"⚠️ 수어 인식 서버 오류: {response_data['error']}")
                        return
                    
                    # 수어 인식 결과 분석
                    sign_prediction = response_data.get('sign_prediction')
                    mediapipe_results = response_data.get('mediapipe_results', {})
                    
                    print(f"📊 수어 인식 결과 분석:")
                    print(f"  - 손 감지: {mediapipe_results.get('hand_detected', False)}")
                    print(f"  - 포즈 감지: {mediapipe_results.get('pose_detected', False)}")
                    
                    if sign_prediction:
                        confidence = sign_prediction.get('confidence', 0)
                        predicted_class = sign_prediction.get('predicted_class', 'Unknown')
                        print(f"  - 예측된 수어: {predicted_class}")
                        print(f"  - 신뢰도: {confidence:.3f}")
                        
                        if confidence >= self.sign_confidence_threshold:
                            print(f"✅ 높은 신뢰도로 수어 인식됨!")
                        else:
                            print(f"⚠️ 신뢰도가 임계값({self.sign_confidence_threshold}) 미만")
                    else:
                        print(f"❌ 수어 예측 결과 없음")
                    
                    # 수어 인식 결과 저장
                    self.last_sign_result = response_data
                    print(f"✅ 수어 인식 결과 업데이트됨!")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ 수어 인식 JSON 파싱 오류: {e}")
                    print(f"응답이 JSON이 아님: {resp.text}")
                except Exception as e:
                    print(f"❌ 수어 인식 서버 응답 파싱 오류: {e}")
            else:
                print(f"❌ 수어 인식 서버 전송 실패: {resp.status_code}")
                
        except Exception as e:
            print(f"❌ 수어 인식 서버 전송 실패: {e}")

    def send_frame_to_server(self, frame):
        """서버로 원본 프레임 전송 (서버에서 YOLO 처리)"""
        self.realtime_fps += 1 
        print(f"self.realtime_fps : {self.realtime_fps}")
        
        # 원본 이미지를 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ 이미지 인코딩 실패")
            return
            
        frame_bytes = buffer.tobytes()
        
        try:
            print(f"📤 서버로 원본 이미지 전송")
            
            files = {'image': (f'{self.realtime_fps}.jpg', frame_bytes, 'image/jpeg')}
            data = {'frame_id': str(self.realtime_fps)}
            
            resp = requests.post(
                "http://192.168.100.135:5000/estimate_pose",
                files=files,
                data=data,
                timeout=10
            )
            
            # 서버 응답 처리 및 포즈 데이터 저장
            if resp.status_code == 200:
                print("✅ 서버 전송 성공")
                try:
                    # 서버 응답 내용 확인
                    print(f"📥 서버 응답 내용: {resp.text[:200]}...")
                    
                    # JSON 파싱
                    response_data = resp.json()
                    print(f"📋 파싱된 응답 키들: {list(response_data.keys())}")
                    
                    # 오류 체크
                    if 'error' in response_data:
                        print(f"⚠️ 서버 오류: {response_data['error']}")
                        return
                    
                    # 키포인트 및 스코어 추출
                    keypoints = response_data.get('keypoints', [])
                    scores = response_data.get('scores', [])
                    person_box = response_data.get('person_box', [])
                    
                    if keypoints and scores:
                        # 키포인트 데이터 분석
                        print(f"📊 키포인트 데이터 분석:")
                        print(f"  - 키포인트 개수: {len(keypoints)}")
                        print(f"  - 스코어 개수: {len(scores)}")
                        
                        # 키포인트 좌표 범위 확인 (이미 원본 이미지 좌표로 변환됨)
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
                        
                        # 검출된 사람 정보
                        if person_box:
                            print(f"  - 검출된 사람 박스: {person_box[:4]} (신뢰도: {person_box[4]:.2f})")
                        
                        self.last_pose_data = {
                            'keypoints': keypoints,
                            'scores': scores,
                            'person_box': person_box
                        }
                        print(f"✅ 포즈 데이터 업데이트됨!")
                    else:
                        print(f"❌ 키포인트 또는 스코어를 찾을 수 없음")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 오류: {e}")
                    print(f"응답이 JSON이 아님: {resp.text}")
                except Exception as e:
                    print(f"❌ 서버 응답 파싱 오류: {e}")
            else:
                print(f"❌ 서버 전송 실패: {resp.status_code}")
                
            self.last_server_result = f"{self.realtime_fps}-서버 결과"
            
        except Exception as e:
            print(f"❌ 실시간 서버 전송 실패: {e}")

    def process_latest_folder_images(self):
        """가장 최근 폴더의 이미지를 서버로 전송"""
        base_dir = "captured_datas"
        if not os.path.exists(base_dir):
            return "저장된 폴더가 없습니다."
        
        folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not folders:
            return "저장된 폴더가 없습니다."
        
        latest_folder = max(folders, key=os.path.getmtime)
        images = sorted(glob.glob(os.path.join(latest_folder, "*.jpg")))
        send_count = 0
        
        for img_path in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            # 이미지를 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"❌ 이미지 인코딩 실패: {img_path}")
                continue
            
            image_bytes = buffer.tobytes()
            
            # 서버로 전송
            try:
                files = {'image': (os.path.basename(img_path), image_bytes, 'image/jpeg')}
                data = {'frame_id': os.path.basename(img_path)}
                
                resp = requests.post(
                    "http://192.168.100.135:5000/estimate_pose",
                    files=files,
                    data=data,
                    timeout=10
                )
                
                if resp.status_code == 200:
                    send_count += 1
                    print(f"✅ 서버로 전송 성공: {send_count}, {img_path}")
                else:
                    print(f"❌ 서버 응답 오류: {resp.status_code} {resp.text}")
                    
            except Exception as e:
                print(f"❌ 서버 전송 실패: {os.path.basename(img_path)} - {e}")
        
        return f"이미지 처리 완료: {len(images)}개 이미지 중 {send_count}개 서버 전송 성공"

    def get_status_info(self):
        """현재 상태 정보 반환"""
        status = {
            'realtime_translate': self.realtime_translate,
            'sign_recognition_mode': self.sign_recognition_mode,
            'show_skeleton': self.show_skeleton,
            'show_sign_recognition': self.show_sign_recognition,
            'recording': self.recording,
            'capturing_images': self.capturing_images,
            'pose_threshold': self.pose_threshold,
            'sign_confidence_threshold': self.sign_confidence_threshold,
            'has_pose_data': self.last_pose_data is not None,
            'has_sign_result': self.last_sign_result is not None,
            'fps': self.fps,
            'frame_size': f"{self.w}x{self.h}"
        }
        
        # 최근 수어 인식 결과 정보 추가
        if self.last_sign_result:
            sign_prediction = self.last_sign_result.get('sign_prediction')
            if sign_prediction:
                status['last_sign'] = {
                    'predicted_class': sign_prediction.get('predicted_class'),
                    'confidence': sign_prediction.get('confidence'),
                    'above_threshold': sign_prediction.get('confidence', 0) >= self.sign_confidence_threshold
                }
            
            mediapipe_results = self.last_sign_result.get('mediapipe_results', {})
            status['mediapipe_status'] = {
                'hand_detected': mediapipe_results.get('hand_detected', False),
                'pose_detected': mediapipe_results.get('pose_detected', False)
            }
        
        return status

    def test_sign_recognition_with_image(self, image_path):
        """특정 이미지로 수어 인식 테스트"""
        if not os.path.exists(image_path):
            return f"이미지 파일을 찾을 수 없습니다: {image_path}"
        
        frame = cv2.imread(image_path)
        if frame is None:
            return f"이미지를 읽을 수 없습니다: {image_path}"
        
        try:
            # 이미지를 JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                return "이미지 인코딩 실패"
                
            frame_bytes = buffer.tobytes()
            
            files = {'image': (os.path.basename(image_path), frame_bytes, 'image/jpeg')}
            data = {
                'frame_id': os.path.basename(image_path),
                'confidence_threshold': str(self.sign_confidence_threshold),
                'include_features': 'false'
            }
            
            resp = requests.post(
                "http://192.168.100.135:5000/recognize_sign",
                files=files,
                data=data,
                timeout=15
            )
            
            if resp.status_code == 200:
                response_data = resp.json()
                
                # 결과 저장
                self.last_sign_result = response_data
                
                # 결과 정보 구성
                result_info = {
                    'success': True,
                    'image_path': image_path,
                    'mediapipe_results': response_data.get('mediapipe_results', {}),
                    'sign_prediction': response_data.get('sign_prediction'),
                    'processing_times': response_data.get('processing_times', {})
                }
                
                return result_info
            else:
                return {
                    'success': False,
                    'error': f"서버 오류: {resp.status_code}",
                    'response': resp.text[:200]
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"요청 실패: {str(e)}"
            }

# 웹캠 제어를 위한 추가 함수들
def print_controls():
    """조작법 출력"""
    print("\n" + "="*60)
    print("🎥 향상된 웹캠 제어 시스템")
    print("="*60)
    print("기본 조작법:")
    print("  - 실시간 번역: start_realtime() / stop_realtime()")
    print("  - 수어 인식 모드: toggle_sign_recognition_mode()")
    print("  - 스켈레톤 표시: toggle_skeleton_display()")
    print("  - 수어 결과 표시: toggle_sign_recognition_display()")
    print("")
    print("설정 조정:")
    print("  - 포즈 임계값: set_pose_threshold(값)")
    print("  - 수어 신뢰도 임계값: set_sign_confidence_threshold(값)")
    print("")
    print("녹화/캡처:")
    print("  - 비디오 녹화: start_recording() / stop_recording()")
    print("  - 이미지 캡처: start_capture_images() / stop_capture_images()")
    print("")
    print("테스트:")
    print("  - 이미지 수어 인식 테스트: test_sign_recognition_with_image('경로')")
    print("  - 상태 확인: get_status_info()")
    print("="*60)

# 사용 예시
if __name__ == "__main__":
    # 웹캠 캡처 객체 생성
    webcam = WebcamCapture()
    
    print_controls()
    
    # 예시 사용법
    print("\n🚀 웹캠 시스템 초기화 완료!")
    print("다음과 같이 사용하세요:")
    print("webcam.toggle_sign_recognition_mode()  # 수어 인식 모드 켜기")
    print("webcam.toggle_skeleton_display()       # 스켈레톤 표시 켜기")
    print("webcam.start_realtime()               # 실시간 처리 시작")
    
    # 테스트를 위한 간단한 실행
    try:
        print("\n📹 웹캠 테스트 시작 (5초간)")
        for i in range(50):  # 약 5초간 테스트
            frame = webcam.capture_frame()
            if frame is not None:
                # 여기서 cv2.imshow()를 사용하여 화면에 표시할 수 있습니다
                # cv2.imshow('Webcam', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                pass
            time.sleep(0.1)
        
        print("✅ 웹캠 테스트 완료")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    
    finally:
        webcam.release_camera()
        print("🔧 리소스 정리 완료")