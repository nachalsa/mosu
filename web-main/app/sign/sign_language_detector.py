import cv2
import numpy as np
import mediapipe as mp
import os
from typing import Optional, Tuple
from collections import deque

try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    print("⚠️ OpenVINO가 설치되지 않았습니다. pip install openvino 실행하세요.")
    OPENVINO_AVAILABLE = False

class SignLanguageDetector:
    """OpenVINO 기반 ASL 수화 인식 모델"""
    
    def __init__(self, model_path: str = None):
        print("🔄 OpenVINO ASL 모델 초기화 중...")
        
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # 시각화용
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 비디오 시퀀스 버퍼 (16프레임)
        self.frame_sequence = deque(maxlen=16)
        
        # OpenVINO 모델 로드
        self.model, self.input_layer, self.output_layer = self._load_openvino_model(model_path)
        
        # ASL 라벨
        self.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        print("✅ OpenVINO ASL 모델 초기화 완료")
    
    def _load_openvino_model(self, model_path: str = None):
        """OpenVINO 모델 로드"""
        try:
            if not OPENVINO_AVAILABLE:
                print("⚠️ OpenVINO 없음, 룰 기반 인식 사용")
                return None, None, None
            
            # 모델 파일 경로 설정
            if model_path is None:
                model_xml = "models/asl_model.xml"
                model_bin = "models/asl_model.bin"
            else:
                model_xml = model_path
                model_bin = model_path.replace('.xml', '.bin')
            
            if not os.path.exists(model_xml) or not os.path.exists(model_bin):
                print(f"⚠️ OpenVINO 모델 파일이 없습니다: {model_xml}")
                return None, None, None
            
            # OpenVINO Core 초기화
            ie = Core()
            
            # 모델 로드
            model = ie.read_model(model=model_xml, weights=model_bin)
            compiled_model = ie.compile_model(model=model, device_name="CPU")
            
            # 입력/출력 레이어 정보 가져오기
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            
            print(f"✅ OpenVINO 모델 로드 성공: {model_xml}")
            print(f"📐 입력 형태: {input_layer.shape}")
            print(f"📐 출력 형태: {output_layer.shape}")
            
            return compiled_model, input_layer, output_layer
            
        except Exception as e:
            print(f"⚠️ OpenVINO 모델 로드 실패: {e}")
            return None, None, None
    
    def _preprocess_image_sequence(self, image: np.ndarray) -> np.ndarray:
        """비디오 시퀀스용 이미지 전처리"""
        try:
            if self.input_layer:
                input_shape = self.input_layer.shape
                print(f"🔍 모델 입력 형태: {input_shape}")
                
                # [N, C, T, H, W] 형태 (비디오 시퀀스)
                if len(input_shape) == 5:
                    n, c, t, h, w = input_shape
                    
                    # 이미지 전처리
                    processed = cv2.resize(image, (w, h))
                    processed = processed.astype(np.float32) / 255.0
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    processed = np.transpose(processed, (2, 0, 1))  # [C, H, W]
                    
                    # 시퀀스 버퍼에 추가
                    self.frame_sequence.append(processed)
                    
                    # 16프레임이 모이면 시퀀스 생성
                    if len(self.frame_sequence) == t:
                        sequence = np.stack(list(self.frame_sequence), axis=1)  # [C, T, H, W]
                        sequence = np.expand_dims(sequence, axis=0)  # [N, C, T, H, W]
                        return sequence
                    else:
                        # 프레임이 부족하면 현재 프레임을 반복해서 16개 채움
                        while len(self.frame_sequence) < t:
                            self.frame_sequence.append(processed)
                        sequence = np.stack(list(self.frame_sequence), axis=1)
                        sequence = np.expand_dims(sequence, axis=0)
                        return sequence
                        
                # [N, C, H, W] 형태 (단일 이미지) - 백업
                elif len(input_shape) == 4:
                    processed = cv2.resize(image, (input_shape[3], input_shape[2]))
                    processed = processed.astype(np.float32) / 255.0
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    processed = np.transpose(processed, (2, 0, 1))
                    processed = np.expand_dims(processed, axis=0)
                    return processed
            
            # 기본 전처리
            processed = cv2.resize(image, (224, 224))
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)
            return processed
            
        except Exception as e:
            print(f"전처리 오류: {e}")
            return None
    
    def _predict_with_openvino(self, image: np.ndarray) -> str:
        """OpenVINO 모델로 예측"""
        try:
            if not self.model:
                return "NO_MODEL"
            
            # 비디오 시퀀스 전처리
            input_data = self._preprocess_image_sequence(image)
            
            if input_data is None:
                return "PREPROCESS_ERROR"
            
            print(f"🔍 입력 데이터 형태: {input_data.shape}")
            
            # 모델 추론
            result = self.model([input_data])[self.output_layer]
            
            # 결과 처리
            predictions = result[0]  # 첫 번째 배치
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            
            print(f"🔍 OpenVINO 예측: 클래스={predicted_class}, 신뢰도={confidence:.3f}")
            
            if confidence > 0.3:
                return self.labels[predicted_class % len(self.labels)]
            else:
                return "UNCERTAIN"
                
        except Exception as e:
            print(f"OpenVINO 예측 오류: {e}")
            return "PREDICTION_ERROR"
    
    def _predict_with_mediapipe_rules(self, image: np.ndarray) -> str:
        """MediaPipe 룰 기반 백업 예측"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                
                # 간단한 제스처 분류
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]
                
                # 손가락 펴진 상태 확인
                fingers_up = []
                
                # 엄지 (x축 기준)
                if thumb_tip.x > landmarks[3].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
                
                # 나머지 손가락들 (y축 기준)
                finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
                finger_pips = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
                
                for tip, pip in zip(finger_tips, finger_pips):
                    if tip.y < pip.y:
                        fingers_up.append(1)
                    else:
                        fingers_up.append(0)
                
                # 제스처 분류
                fingers_count = sum(fingers_up)
                
                if fingers_count == 0:
                    return "A"  # 주먹
                elif fingers_count == 1 and fingers_up[1] == 1:
                    return "D"  # 검지만
                elif fingers_count == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
                    return "V"  # 브이사인
                elif fingers_count == 5:
                    return "B"  # 펼친 손
                else:
                    return f"GESTURE_{fingers_count}"
            else:
                return "NO_HANDS"
                
        except Exception as e:
            print(f"룰 기반 예측 오류: {e}")
            return "RULE_ERROR"
    
    def predict_sign_with_visualization(self, image: np.ndarray) -> Tuple[str, np.ndarray]:
        """수화 예측 + 시각화"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hands_results = self.hands.process(rgb_image)
            
            # 원본 이미지 복사 (시각화용)
            annotated_image = image.copy()
            
            sign_result = "NO_HANDS"
            
            if hands_results.multi_hand_landmarks:
                # 손 뼈대 그리기
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # OpenVINO 모델로 예측 (우선) - 실시간에서는 룰 기반 사용
                # if self.model:
                #     sign_result = self._predict_with_openvino(image)
                #     method = "OpenVINO"
                # else:
                # 실시간 처리를 위해 룰 기반 사용 (더 빠름)
                sign_result = self._predict_with_mediapipe_rules(image)
                method = "Rules"
                
                # 결과 텍스트 표시
                cv2.putText(
                    annotated_image, 
                    f"ASL ({method}): {sign_result}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2
                )
            
            return sign_result, annotated_image
            
        except Exception as e:
            print(f"시각화 수화 예측 오류: {e}")
            return "ERROR", image
    
    def predict_sign(self, image: np.ndarray) -> str:
        """간단한 수화 예측 (시각화 없음)"""
        result, _ = self.predict_sign_with_visualization(image)
        return result