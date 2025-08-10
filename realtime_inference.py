#!/usr/bin/env python3
"""
실시간 수화 인식 추론기
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Deque
import logging

from sign_language_model import SequenceToSequenceSignModel, RealtimeDecoder

logger = logging.getLogger(__name__)

class AdaptiveWindow:
    """적응형 슬라이딩 윈도우"""
    
    def __init__(self, 
                 max_window: int = 120,    # 4초 (30fps 기준)
                 min_window: int = 15,     # 0.5초 
                 overlap: int = 30):       # 1초 오버랩
        self.max_window = max_window
        self.min_window = min_window
        self.overlap = overlap
        
        self.buffer: Deque[np.ndarray] = deque(maxlen=max_window)
        self.frame_count = 0
        
    def add_frame(self, pose_features: np.ndarray):
        """프레임 추가"""
        self.buffer.append(pose_features)
        self.frame_count += 1
        
    def should_process(self) -> bool:
        """처리 여부 판단"""
        return len(self.buffer) >= self.min_window
    
    def get_window(self) -> np.ndarray:
        """현재 윈도우 반환"""
        if len(self.buffer) == 0:
            return np.zeros((self.min_window, 133, 3), dtype=np.float32)
        
        window_data = np.array(list(self.buffer))  # [frames, 133, 3]
        
        if len(window_data) < self.min_window:
            # 패딩 추가
            padding = np.zeros((self.min_window - len(window_data), 133, 3), dtype=np.float32)
            window_data = np.concatenate([padding, window_data], axis=0)
        
        return window_data
    
    def reset(self):
        """버퍼 리셋"""
        self.buffer.clear()
        self.frame_count = 0

class PoseFeatureExtractor:
    """포즈 특징 추출기 (더미 - MediaPipe 등으로 대체 필요)"""
    
    def __init__(self):
        self.n_keypoints = 133
        
    def extract_pose_features(self, frame: np.ndarray) -> np.ndarray:
        """프레임에서 포즈 특징 추출"""
        # 실제로는 MediaPipe나 다른 포즈 추정 모델 사용
        # 여기서는 더미 데이터 생성
        
        h, w = frame.shape[:2]
        
        # 더미 키포인트 (실제로는 MediaPipe 결과 사용)
        keypoints = np.random.rand(self.n_keypoints, 2) * np.array([w, h])  # [133, 2]
        scores = np.random.rand(self.n_keypoints) * 10  # [133]
        
        # [133, 3] 형태로 결합 (x, y, score)
        pose_features = np.zeros((self.n_keypoints, 3), dtype=np.float32)
        pose_features[:, :2] = keypoints
        pose_features[:, 2] = scores
        
        return pose_features

class RealTimeSignLanguageInference:
    """실시간 수화 인식 추론기"""
    
    def __init__(self,
                 model_path: str,
                 vocab_path: str = None,
                 device: str = "cuda",
                 confidence_threshold: float = 0.7,
                 boundary_threshold: float = 0.8):
        
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.boundary_threshold = boundary_threshold
        
        # 모델 로드
        self._load_model(model_path)
        
        # Vocabulary 로드
        self._load_vocabulary(vocab_path)
        
        # 컴포넌트 초기화
        self.window = AdaptiveWindow()
        self.pose_extractor = PoseFeatureExtractor()
        self.decoder = RealtimeDecoder(
            vocab_size=len(self.words),
            confidence_threshold=confidence_threshold,
            boundary_threshold=boundary_threshold
        )
        
        # 통계
        self.frame_count = 0
        self.inference_times = deque(maxlen=100)
        self.detected_words = []
        
        logger.info(f"✅ 실시간 추론기 초기화 완료")
        logger.info(f"   - 디바이스: {self.device}")
        logger.info(f"   - Vocabulary 크기: {len(self.words)}")
        logger.info(f"   - 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 생성 (체크포인트에서 설정 추출)
        vocab_size = len(checkpoint.get('vocab_words', []))
        if vocab_size == 0:
            vocab_size = 442  # 기본값
        
        self.model = SequenceToSequenceSignModel(
            vocab_size=vocab_size,
            embed_dim=256,
            num_encoder_layers=6,
            num_decoder_layers=4,
            num_heads=8
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"📚 모델 로드 완료: {model_path}")
    
    def _load_vocabulary(self, vocab_path: str = None):
        """Vocabulary 로드"""
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.words = vocab_data['words']
            self.word_to_id = vocab_data['word_to_id']
        else:
            # 기본 vocabulary (더미)
            self.words = [f"단어_{i:03d}" for i in range(442)]
            self.word_to_id = {word: i for i, word in enumerate(self.words)}
        
        logger.info(f"📖 Vocabulary 로드: {len(self.words)}개 단어")
    
    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """프레임 처리 및 추론"""
        self.frame_count += 1
        
        # 1. 포즈 특징 추출
        pose_features = self.pose_extractor.extract_pose_features(frame)  # [133, 3]
        
        # 2. 윈도우에 추가
        self.window.add_frame(pose_features)
        
        # 3. 처리 가능한지 확인
        if not self.window.should_process():
            return None
        
        # 4. 모델 추론
        start_time = time.time()
        
        # 윈도우 데이터 준비
        window_data = self.window.get_window()  # [frames, 133, 3]
        
        # 배치 차원 추가 및 텐서 변환
        input_tensor = torch.from_numpy(window_data).unsqueeze(0).to(self.device)  # [1, frames, 133, 3]
        
        # 모델 추론
        outputs = self.model(input_tensor)
        
        # 현재 프레임 (마지막 프레임)의 출력 사용
        current_frame_idx = -1
        word_logits = outputs['word_logits'][0, current_frame_idx]  # [vocab_size]
        boundary_logits = outputs['boundary_logits'][0, current_frame_idx]  # [3]
        confidence_score = outputs['confidence_scores'][0, current_frame_idx]  # scalar
        
        # 5. 실시간 디코더 처리
        detected_word_id = self.decoder.process_frame_output(
            word_logits, boundary_logits, confidence_score
        )
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 6. 결과 처리
        if detected_word_id is not None:
            detected_word = self.words[detected_word_id]
            self.detected_words.append({
                'word': detected_word,
                'word_id': detected_word_id,
                'frame': self.frame_count,
                'timestamp': time.time(),
                'confidence': confidence_score.item()
            })
            
            logger.info(f"🎯 단어 검출: '{detected_word}' (신뢰도: {confidence_score:.3f})")
            return detected_word
        
        return None
    
    def get_statistics(self) -> Dict:
        """성능 통계 반환"""
        if len(self.inference_times) == 0:
            avg_fps = 0
        else:
            avg_inference_time = np.mean(self.inference_times)
            avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'frame_count': self.frame_count,
            'detected_words_count': len(self.detected_words),
            'avg_inference_fps': avg_fps,
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0,
            'decoder_state': self.decoder.state,
            'buffer_size': len(self.window.buffer)
        }
    
    def reset(self):
        """상태 리셋"""
        self.window.reset()
        self.decoder.reset_state()
        self.frame_count = 0
        self.detected_words = []
        self.inference_times.clear()
        
        logger.info("🔄 추론기 상태 리셋")

class WebcamInferenceDemo:
    """웹캠 실시간 추론 데모"""
    
    def __init__(self, 
                 model_path: str,
                 camera_id: int = 0,
                 display_width: int = 1280,
                 display_height: int = 720):
        
        self.camera_id = camera_id
        self.display_width = display_width
        self.display_height = display_height
        
        # 추론기 초기화
        self.inference = RealTimeSignLanguageInference(model_path)
        
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다: {camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # UI 상태
        self.recent_words = deque(maxlen=5)
        self.show_stats = True
        
        logger.info(f"🎥 웹캠 데모 초기화 완료: {display_width}x{display_height}")
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """UI 요소 그리기"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # 반투명 패널 배경
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        # 제목
        cv2.putText(vis_frame, "Real-Time Sign Language Recognition", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 최근 검출된 단어들
        recent_text = "Recent Words: " + " -> ".join(self.recent_words)
        cv2.putText(vis_frame, recent_text,
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 통계 표시
        if self.show_stats:
            stats = self.inference.get_statistics()
            stats_text = f"FPS: {stats['avg_inference_fps']:.1f} | " \
                        f"Frame: {stats['frame_count']} | " \
                        f"Detected: {stats['detected_words_count']} | " \
                        f"State: {stats['decoder_state']}"
            cv2.putText(vis_frame, stats_text,
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 키 안내
        cv2.putText(vis_frame, "Press 'q' to quit, 'r' to reset, 's' to toggle stats",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis_frame
    
    def run(self):
        """메인 루프 실행"""
        logger.info("🚀 웹캠 데모 시작 - 'q'로 종료")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("프레임 읽기 실패")
                    break
                
                # 추론 실행
                detected_word = self.inference.process_frame(frame)
                
                # 검출된 단어 처리
                if detected_word:
                    self.recent_words.append(detected_word)
                    print(f"🎯 검출: {detected_word}")
                
                # UI 그리기
                vis_frame = self.draw_ui(frame)
                
                # 화면 표시
                cv2.imshow("Sign Language Recognition", vis_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.inference.reset()
                    self.recent_words.clear()
                    print("🔄 리셋")
                elif key == ord('s'):
                    self.show_stats = not self.show_stats
                    print(f"📊 통계 표시: {self.show_stats}")
        
        except KeyboardInterrupt:
            logger.info("키보드 인터럽트로 종료")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리"""
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 최종 통계
        stats = self.inference.get_statistics()
        logger.info("📊 최종 통계:")
        logger.info(f"   총 프레임: {stats['frame_count']}")
        logger.info(f"   검출된 단어: {stats['detected_words_count']}")
        logger.info(f"   평균 FPS: {stats['avg_inference_fps']:.1f}")
        
        # 검출된 단어 목록
        if self.inference.detected_words:
            logger.info("🎯 검출된 단어들:")
            for word_info in self.inference.detected_words:
                logger.info(f"   {word_info['word']} (프레임 {word_info['frame']}, 신뢰도 {word_info['confidence']:.3f})")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="실시간 수화 인식 추론")
    parser.add_argument("--model", type=str, required=True,
                       help="모델 체크포인트 경로")
    parser.add_argument("--vocab", type=str, default=None,
                       help="Vocabulary JSON 파일 경로")
    parser.add_argument("--camera", type=int, default=0,
                       help="카메라 ID (기본: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="추론 디바이스")
    parser.add_argument("--width", type=int, default=1280,
                       help="화면 너비")
    parser.add_argument("--height", type=int, default=720,
                       help="화면 높이")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="신뢰도 임계값")
    
    args = parser.parse_args()
    
    print("🚀 실시간 수화 인식 데모")
    print("=" * 40)
    print(f"모델: {args.model}")
    print(f"카메라: {args.camera}")
    print(f"디바이스: {args.device}")
    print(f"화면 크기: {args.width}x{args.height}")
    print("=" * 40)
    
    try:
        # 웹캠 데모 실행
        demo = WebcamInferenceDemo(
            model_path=args.model,
            camera_id=args.camera,
            display_width=args.width,
            display_height=args.height
        )
        demo.run()
        
    except Exception as e:
        logger.error(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
