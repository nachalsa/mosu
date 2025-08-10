
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

class SignLanguageDetector:
    def _predict_with_mediapipe_rules(self, image: np.ndarray) -> str:
        """MediaPipe 손 랜드마크 기반 간단한 rule-based 수화 인식 (예시: 엄지, 검지 등 펴진 손가락 개수)"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            if not results.multi_hand_landmarks:
                return "NO_HANDS"
            # 예시: 한 손만 인식, 펴진 손가락 개수로 단순 분류
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            # 손가락 tip 인덱스: [4, 8, 12, 16, 20] (엄지~새끼)
            tips = [4, 8, 12, 16, 20]
            fingers_up = []
            # 엄지: x축, 나머지: y축
            if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
            for tip in tips[1:]:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            up_count = sum(fingers_up)
            # 간단한 rule: 손가락 개수로 분류
            if up_count == 0:
                return "FIST"
            elif up_count == 1:
                return "ONE"
            elif up_count == 2:
                return "TWO"
            elif up_count == 3:
                return "THREE"
            elif up_count == 4:
                return "FOUR"
            elif up_count == 5:
                return "FIVE"
            else:
                return f"{up_count}_FINGERS"
        except Exception as e:
            print(f"MediaPipe rule 예측 오류: {e}")
            return "ERROR"
    """MediaPipe 룰 기반 ASL 수화 인식기"""
    
    def __init__(self, model_path: str = None):
        print("🔄 MediaPipe 룰 기반 수화 인식기 초기화 중...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        print("✅ MediaPipe 룰 기반 수화 인식기 초기화 완료")
    
    # OpenVINO 관련 메서드 완전 제거
    
    # _preprocess_image_sequence 등 OpenVINO 관련 메서드 완전 제거
    def predict_sign_with_visualization(self, image: np.ndarray) -> Tuple[str, np.ndarray]:
        """MediaPipe 룰 기반 수화 예측 + 시각화"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hands_results = self.hands.process(rgb_image)
            annotated_image = image.copy()
            sign_result = "NO_HANDS"
            method = "Rules"
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                sign_result = self._predict_with_mediapipe_rules(image)
                method = "Rules"
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
        """MediaPipe 룰 기반 수화 예측 (시각화 없음)"""
        return self._predict_with_mediapipe_rules(image)