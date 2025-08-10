#!/usr/bin/env python3
"""
정규화 상수 (범위 매핑 방식: 표준범위 → 0.1~0.9)
KEYPOINT_NORM = {
    'x_min': 0.0,      # 표준 최소값
    'x_max': 2304.0,   # 표준 최대값 
    'y_min': 0.0,      # 표준 최소값
    'y_max': 3072.0,   # 표준 최대값
    'target_min': 0.1, # 목표 최소값
    'target_max': 0.9, # 목표 최대값
}
SCORE_NORM = {
    'min': 0.0,
    'max': 10.0,       # 표준 최대값
    'target_min': 0.0,
    'target_max': 1.0,
}테이션 데이터 로더
- 포즈 데이터: batch_SEN_XX_YY_F_poses.h5 (keypoints + scores)
- 어노테이션: sign_language_dataset_only_sen_lzf.h5 (세그먼트 정보)

정규화 정보:
- keypoints_scaled: x(0~2304), y(0~3072) → 정규화: /2304, /3072
- scores: 0.0~10.0 → 정규화: /10.0
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import random
import math

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 정규화 상수 (범위 매핑 방식: 표준범위 → 0.1~0.9)
KEYPOINT_NORM = {
    'x_min': 0.0,      # 표준 최소값
    'x_max': 2304.0,   # 표준 최대값 
    'y_min': 0.0,      # 표준 최소값
    'y_max': 3072.0,   # 표준 최대값
    'target_min': 0.1, # 목표 최소값
    'target_max': 0.9, # 목표 최대값
}
SCORE_NORM = {
    'min': 0.0,
    'max': 10.0,       # 표준 최대값
    'target_min': 0.0,
    'target_max': 1.0,
}

class PoseDataAugmentation:
    """수화 포즈 데이터 증강 클래스"""
    
    def __init__(self, 
                 enable_horizontal_flip: bool = True,
                 enable_rotation: bool = True,
                 enable_scaling: bool = True,
                 enable_noise: bool = True,
                 horizontal_flip_prob: float = 0.5,
                 rotation_range: float = 15.0,  # ±15도
                 scaling_range: Tuple[float, float] = (0.9, 1.1),  # 90%~110%
                 noise_std: float = 0.01):  # 노이즈 표준편차
        """
        Args:
            enable_horizontal_flip: 좌우 반전 활성화
            enable_rotation: 회전 증강 활성화
            enable_scaling: 크기 증강 활성화
            enable_noise: 노이즈 증강 활성화
            horizontal_flip_prob: 좌우 반전 확률
            rotation_range: 회전 각도 범위 (도)
            scaling_range: 크기 변환 범위
            noise_std: 가우시안 노이즈 표준편차
        """
        self.enable_horizontal_flip = enable_horizontal_flip
        self.enable_rotation = enable_rotation
        self.enable_scaling = enable_scaling
        self.enable_noise = enable_noise
        
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.noise_std = noise_std
        
        logger.info(f"🎨 데이터 증강 설정:")
        logger.info(f"   - 좌우 반전: {'✅' if enable_horizontal_flip else '❌'} (확률: {horizontal_flip_prob})")
        logger.info(f"   - 회전: {'✅' if enable_rotation else '❌'} (±{rotation_range}°)")
        logger.info(f"   - 크기 변환: {'✅' if enable_scaling else '❌'} ({scaling_range[0]:.1f}x~{scaling_range[1]:.1f}x)")
        logger.info(f"   - 노이즈: {'✅' if enable_noise else '❌'} (std: {noise_std})")
    
    def __call__(self, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 증강 적용
        
        Args:
            keypoints: [frames, 133, 2] - 정규화된 키포인트 (0~1)
            scores: [frames, 133] - 정규화된 스코어 (0~1)
            
        Returns:
            augmented_keypoints, augmented_scores
        """
        augmented_keypoints = keypoints.copy()
        augmented_scores = scores.copy()
        
        # 1. 좌우 반전
        if self.enable_horizontal_flip and random.random() < self.horizontal_flip_prob:
            augmented_keypoints = self._horizontal_flip(augmented_keypoints)
        
        # 2. 회전 (중심점 기준)
        if self.enable_rotation:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            augmented_keypoints = self._rotate(augmented_keypoints, angle)
        
        # 3. 크기 변환 (중심점 기준)
        if self.enable_scaling:
            scale = random.uniform(self.scaling_range[0], self.scaling_range[1])
            augmented_keypoints = self._scale(augmented_keypoints, scale)
        
        # 4. 가우시안 노이즈 추가
        if self.enable_noise:
            augmented_keypoints = self._add_noise(augmented_keypoints)
            # 스코어에도 약간의 노이즈 (신뢰도 변화 시뮬레이션)
            augmented_scores = self._add_score_noise(augmented_scores)
        
        # 정규화 범위 유지 (0~1)
        augmented_keypoints = np.clip(augmented_keypoints, 0.0, 1.0)
        augmented_scores = np.clip(augmented_scores, 0.0, 1.0)
        
        return augmented_keypoints, augmented_scores
    
    def _horizontal_flip(self, keypoints: np.ndarray) -> np.ndarray:
        """좌우 반전 (X 좌표 뒤집기)"""
        flipped = keypoints.copy()
        flipped[:, :, 0] = 1.0 - flipped[:, :, 0]  # X 좌표 반전
        return flipped
    
    def _rotate(self, keypoints: np.ndarray, angle_degrees: float) -> np.ndarray:
        """중심점 기준 회전"""
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated = keypoints.copy()
        
        # 중심점을 0.5, 0.5로 가정 (정규화된 좌표계에서)
        center_x, center_y = 0.5, 0.5
        
        # 중심점 기준으로 이동
        rotated[:, :, 0] -= center_x
        rotated[:, :, 1] -= center_y
        
        # 회전 변환 적용
        x_rotated = rotated[:, :, 0] * cos_a - rotated[:, :, 1] * sin_a
        y_rotated = rotated[:, :, 0] * sin_a + rotated[:, :, 1] * cos_a
        
        # 다시 원래 위치로 이동
        rotated[:, :, 0] = x_rotated + center_x
        rotated[:, :, 1] = y_rotated + center_y
        
        return rotated
    
    def _scale(self, keypoints: np.ndarray, scale_factor: float) -> np.ndarray:
        """중심점 기준 크기 변환"""
        scaled = keypoints.copy()
        
        # 중심점을 0.5, 0.5로 가정
        center_x, center_y = 0.5, 0.5
        
        # 중심점 기준으로 이동
        scaled[:, :, 0] -= center_x
        scaled[:, :, 1] -= center_y
        
        # 크기 변환 적용
        scaled[:, :, 0] *= scale_factor
        scaled[:, :, 1] *= scale_factor
        
        # 다시 원래 위치로 이동
        scaled[:, :, 0] += center_x
        scaled[:, :, 1] += center_y
        
        return scaled
    
    def _add_noise(self, keypoints: np.ndarray) -> np.ndarray:
        """가우시안 노이즈 추가"""
        noise = np.random.normal(0, self.noise_std, keypoints.shape).astype(np.float32)
        return keypoints + noise
    
    def _add_score_noise(self, scores: np.ndarray) -> np.ndarray:
        """스코어에 노이즈 추가 (신뢰도 변화 시뮬레이션)"""
        noise = np.random.normal(0, self.noise_std * 0.5, scores.shape).astype(np.float32)
        return scores + noise

class UnifiedSignLanguageDataset(Dataset):
    """통합 수화 데이터셋 (포즈 + 어노테이션)"""
    
    def __init__(self, 
                 annotation_path: str = "./data/sign_language_dataset_only_sen_lzf.h5",
                 pose_data_dir: str = "./data",
                 sequence_length: int = 200,  # 최대 시퀀스 길이
                 min_segment_length: int = 10,  # 최소 세그먼트 길이
                 max_segment_length: int = 300,  # 최대 세그먼트 길이
                 enable_augmentation: bool = False,  # 데이터 증강 활성화
                 augmentation_config: Dict = None):  # 증강 설정
        
        self.annotation_path = annotation_path
        self.pose_data_dir = Path(pose_data_dir)
        self.sequence_length = sequence_length
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.enable_augmentation = enable_augmentation
        
        # 데이터 증강 설정
        if enable_augmentation:
            aug_config = augmentation_config or {}
            self.augmentation = PoseDataAugmentation(**aug_config)
            logger.info("✅ 데이터 증강 활성화됨")
        else:
            self.augmentation = None
            logger.info("❌ 데이터 증강 비활성화됨")
        
        # 데이터 로드
        self._load_annotations()
        self._load_pose_file_mapping()  # 포즈 파일을 먼저 스캔
        self._filter_valid_segments()   # 포즈 데이터가 있는 세그먼트만 필터링
        
        logger.info(f"✅ 데이터셋 로드 완료: {len(self.valid_segments)}개 유효 세그먼트")
        logger.info(f"   - Vocabulary 크기: {self.vocab_size}")
        logger.info(f"   - 평균 세그먼트 길이: {np.mean([s['duration'] for s in self.valid_segments]):.1f} 프레임")
        logger.info(f"   - 데이터 증강: {'✅ 활성화' if enable_augmentation else '❌ 비활성화'}")
        
    def _load_annotations(self):
        """어노테이션 데이터 로드"""
        logger.info(f"📋 어노테이션 로드 중: {self.annotation_path}")
        
        with h5py.File(self.annotation_path, 'r') as f:
            # Vocabulary 로드
            words_data = f['vocabulary']['words'][:]
            # 단어 디코딩 (bytes → string)
            self.words = []
            for w in words_data:
                if isinstance(w, bytes):
                    self.words.append(w.decode('utf-8'))
                else:
                    self.words.append(str(w))
            
            self.vocab_size = len(self.words)
            self.word_to_id = {word: idx for idx, word in enumerate(self.words)}
            self.vocab = {idx: word for idx, word in enumerate(self.words)}  # ID → 단어 매핑 추가
            self.fps = 30  # 기본값
            
            logger.info(f"   Vocabulary 크기: {self.vocab_size}")
            logger.info(f"   첫 5개 단어: {self.words[:5]}")
            
            # 세그먼트 정보 로드
            n_segments = len(f['segments']['data_types'])
            self.segments = []
            
            for i in range(n_segments):
                vocab_len = int(f['segments']['vocab_lens'][i])
                segment = {
                    'index': i,
                    'data_type': int(f['segments']['data_types'][i]),  # 1=SEN (모든 데이터가 SEN)
                    'data_id': int(f['segments']['data_ids'][i]),
                    'real_id': int(f['segments']['real_ids'][i]),
                    'view': int(f['segments']['views'][i]),  # 0=F (정면만 사용)
                    'start_frame': int(f['segments']['start_frames'][i]),
                    'end_frame': int(f['segments']['end_frames'][i]),
                    'duration': int(f['segments']['duration_frames'][i]),
                    'vocab_len': vocab_len,
                    'vocab_ids': f['segments']['vocab_ids'][i, :vocab_len].tolist()
                }
                self.segments.append(segment)
    
    def _filter_valid_segments(self):
        """유효한 세그먼트 필터링 - 포즈 데이터가 있는 세그먼트만"""
        logger.info("🔍 유효한 세그먼트 필터링 중...")
        
        # 포즈 데이터가 있는 Real ID들 확인
        available_real_ids = set(real_id for real_id, _ in self.pose_files.keys())
        logger.info(f"   포즈 데이터가 있는 Real IDs: {sorted(available_real_ids)}")
        
        self.valid_segments = []
        filtered_counts = {
            'total': 0,
            'sen_type': 0,
            'front_view': 0,
            'proper_length': 0,
            'has_pose_data': 0,
            'final_valid': 0
        }
        
        for segment in self.segments:
            filtered_counts['total'] += 1
            
            # 조건 1: SEN 타입
            if segment['data_type'] != 1:
                continue
            filtered_counts['sen_type'] += 1
            
            # 조건 2: 정면(F)
            if segment['view'] != 0:
                continue
            filtered_counts['front_view'] += 1
            
            # 조건 3: 적절한 길이
            if not (self.min_segment_length <= segment['duration'] <= self.max_segment_length):
                continue
            filtered_counts['proper_length'] += 1
            
            # 조건 4: 포즈 데이터 존재 여부 (새로 추가된 조건)
            if segment['real_id'] not in available_real_ids:
                continue
            filtered_counts['has_pose_data'] += 1
            
            # 최종 유효 세그먼트
            self.valid_segments.append(segment)
            filtered_counts['final_valid'] += 1
        
        logger.info(f"   필터링 결과:")
        logger.info(f"     총 세그먼트: {filtered_counts['total']}")
        logger.info(f"     SEN 타입: {filtered_counts['sen_type']}")
        logger.info(f"     정면(F): {filtered_counts['front_view']}")
        logger.info(f"     적절한 길이: {filtered_counts['proper_length']}")
        logger.info(f"     포즈 데이터 존재: {filtered_counts['has_pose_data']}")
        logger.info(f"     최종 유효: {filtered_counts['final_valid']}")
        
        # Real ID별 분포 확인
        real_id_distribution = {}
        for segment in self.valid_segments:
            real_id = segment['real_id']
            real_id_distribution[real_id] = real_id_distribution.get(real_id, 0) + 1
        
        logger.info(f"   Real ID별 유효 세그먼트 분포:")
        for real_id in sorted(real_id_distribution.keys()):
            count = real_id_distribution[real_id]
            logger.info(f"     Real ID {real_id:2d}: {count:5d}개")
        
        # 예상 비율 vs 실제 비율
        expected_ratio = len(available_real_ids) / 16  # 16명 중 실제 데이터가 있는 비율
        actual_ratio = filtered_counts['final_valid'] / filtered_counts['proper_length']  # 길이 조건 통과 대비 최종 유효
        logger.info(f"   예상 비율 (포즈 데이터 기준): {expected_ratio:.3f} ({len(available_real_ids)}/16)")
        logger.info(f"   실제 비율: {actual_ratio:.3f} ({filtered_counts['final_valid']}/{filtered_counts['proper_length']})")
    
    def _load_pose_file_mapping(self):
        """포즈 파일 매핑 생성"""
        logger.info("🗂️ 포즈 파일 매핑 생성 중...")
        
        self.pose_files = {}
        pose_pattern = "batch_SEN_{:02d}_{:02d}_F_poses.h5"
        
        # 실제 존재하는 포즈 파일들 스캔
        available_files = {}
        for real_id in range(1, 17):  # Real ID 1-16
            for batch_id in range(8):  # Batch ID 0-7 (일반적으로)
                pose_file = self.pose_data_dir / pose_pattern.format(real_id, batch_id)
                if pose_file.exists():
                    # 파일에서 실제 data_id 범위 확인
                    try:
                        with h5py.File(pose_file, 'r') as f:
                            video_keys = [k for k in f.keys() if k.startswith('video_sen')]
                            if video_keys:
                                # video_sen0001 → 1
                                data_ids = [int(k.replace('video_sen', '')) for k in video_keys]
                                min_id, max_id = min(data_ids), max(data_ids)
                                available_files[(real_id, batch_id)] = {
                                    'file': str(pose_file),
                                    'data_id_range': (min_id, max_id),
                                    'video_keys': set(video_keys)
                                }
                                logger.debug(f"   발견: Real{real_id:02d}_Batch{batch_id:02d} → data_id {min_id}-{max_id}")
                    except Exception as e:
                        logger.warning(f"   파일 스캔 실패: {pose_file.name} - {e}")
        
        # 세그먼트별 매핑
        self.pose_files = available_files
        logger.info(f"   {len(self.pose_files)}개 포즈 파일 매핑 완료")
    
    def _get_pose_data(self, segment: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """세그먼트에 해당하는 포즈 데이터 로드"""
        data_id = segment['data_id']
        
        # 모든 파일에서 해당 data_id를 포함하는 파일 찾기
        target_file = None
        for file_key, file_info in self.pose_files.items():
            video_key = f"video_sen{data_id:04d}"
            if video_key in file_info['video_keys']:
                target_file = file_info['file']
                break
        
        if target_file is None:
            logger.warning(f"⚠️ 세그먼트 {data_id}: 포즈 데이터 파일을 찾을 수 없음")
            return self._create_dummy_pose_data(segment)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with h5py.File(target_file, 'r') as f:
                    # 비디오 키 생성
                    video_key = f"video_sen{data_id:04d}"
                    
                    if video_key not in f:
                        logger.warning(f"⚠️ 비디오 키 '{video_key}'가 파일 {target_file}에 없음")
                        return self._create_dummy_pose_data(segment)
                    
                    # 키포인트와 스코어 로드
                    keypoints = f[video_key]['keypoints_scaled'][:]  # [frames, 133, 2]
                    scores = f[video_key]['scores'][:]  # [frames, 133]
                    
                    # 범위 매핑 정규화 (표준범위 → 0.1~0.9)
                    
                    # X 좌표: 0~2304 → 0.1~0.9 선형 매핑
                    x_norm = (keypoints[:, :, 0] - KEYPOINT_NORM['x_min']) / (KEYPOINT_NORM['x_max'] - KEYPOINT_NORM['x_min'])  # 0~1
                    x_mapped = KEYPOINT_NORM['target_min'] + x_norm * (KEYPOINT_NORM['target_max'] - KEYPOINT_NORM['target_min'])  # 0.1~0.9
                    keypoints[:, :, 0] = np.clip(x_mapped, 0.0, 1.0)  # 이상치 최종 클리핑
                    
                    # Y 좌표: 0~3072 → 0.1~0.9 선형 매핑
                    y_norm = (keypoints[:, :, 1] - KEYPOINT_NORM['y_min']) / (KEYPOINT_NORM['y_max'] - KEYPOINT_NORM['y_min'])  # 0~1
                    y_mapped = KEYPOINT_NORM['target_min'] + y_norm * (KEYPOINT_NORM['target_max'] - KEYPOINT_NORM['target_min'])  # 0.1~0.9
                    keypoints[:, :, 1] = np.clip(y_mapped, 0.0, 1.0)  # 이상치 최종 클리핑
                    
                    # 스코어: 0~10 → 0.0~1.0 선형 매핑
                    s_norm = (scores - SCORE_NORM['min']) / (SCORE_NORM['max'] - SCORE_NORM['min'])  # 0~1
                    s_mapped = SCORE_NORM['target_min'] + s_norm * (SCORE_NORM['target_max'] - SCORE_NORM['target_min'])  # 0.1~0.9
                    scores = np.clip(s_mapped, 0.0, 1.0)  # 이상치 최종 클리핑
                    
                    logger.debug(f"정규화 후 범위 - 키포인트: [{keypoints.min():.3f}, {keypoints.max():.3f}], 스코어: [{scores.min():.3f}, {scores.max():.3f}]")
                    
                    # 세그먼트 범위 추출 - 안전한 범위 조정
                    start_frame = max(0, segment['start_frame'])
                    end_frame = segment['end_frame']
                    actual_frames = keypoints.shape[0]
                    
                    # 유효한 프레임 범위 확인 및 조정
                    if end_frame > actual_frames:
                        logger.debug(f"프레임 범위 조정: {end_frame} → {actual_frames} (Data ID: {data_id})")
                        end_frame = actual_frames
                    
                    # 시작 프레임이 유효 범위를 벗어나는 경우 조정
                    if start_frame >= actual_frames:
                        logger.debug(f"시작 프레임 범위 조정: {start_frame} → {max(0, actual_frames - 1)} (Data ID: {data_id})")
                        start_frame = max(0, actual_frames - 1)
                    
                    # 최소 길이 보장 및 범위 재검증
                    if start_frame >= end_frame:
                        # 최소 1프레임은 확보하되, 더 안정적으로 조정
                        if actual_frames > 0:
                            # 가능한 한 원본 범위에 가깝게 조정
                            min_length = 1
                            if actual_frames >= min_length:
                                start_frame = max(0, min(start_frame, actual_frames - min_length))
                                end_frame = min(actual_frames, start_frame + min_length)
                                logger.debug(f"프레임 범위 수정: [{start_frame}-{end_frame}] (Data ID: {data_id})")
                            else:
                                start_frame = 0
                                end_frame = actual_frames
                                logger.debug(f"전체 프레임 사용: [0-{actual_frames}] (Data ID: {data_id})")
                        else:
                            logger.debug(f"빈 데이터 생성 (Data ID: {data_id})")
                            return self._create_dummy_pose_data(segment)
                    
                    # 너무 짧은 세그먼트 처리 (성능 개선)
                    segment_length = end_frame - start_frame
                    if segment_length < 3:
                        # 가능한 범위에서 확장
                        center = (start_frame + end_frame) // 2
                        half_length = max(1, segment_length // 2)
                        start_frame = max(0, center - half_length - 1)
                        end_frame = min(actual_frames, center + half_length + 2)
                        if segment_length == 1:  # 1프레임 세그먼트만 로그
                            logger.debug(f"1프레임 세그먼트 확장: Data ID {data_id}")
                    
                    # 세그먼트 추출
                    segment_keypoints = keypoints[start_frame:end_frame]  # [seg_frames, 133, 2]
                    segment_scores = scores[start_frame:end_frame]  # [seg_frames, 133]
                    
                    # 최종 데이터 검증 (간소화)
                    if segment_keypoints.shape[0] == 0 or segment_keypoints.shape[1] != 133:
                        logger.warning(f"⚠️ 잘못된 포즈 데이터 형태: {segment_keypoints.shape} (Data ID: {data_id})")
                        return None  # 더미 데이터 대신 None 반환
                    
                    return segment_keypoints, segment_scores
                    
            except (OSError, IOError) as e:
                logger.warning(f"파일 I/O 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1)  # 짧은 대기
                    continue
            except Exception as e:
                logger.error(f"포즈 데이터 로드 실패: {e}")
                break
        
        # 모든 재시도 실패 시 None 반환 (더미 데이터 사용하지 않음)
        logger.warning(f"⚠️ 포즈 데이터 로드 완전 실패 (Data ID: {data_id})")
        return None
    
    def _create_dummy_pose_data(self, segment):
        """더미 포즈 데이터 생성 (정규화된 범위)"""
        duration = segment['end_frame'] - segment['start_frame']
        duration = max(1, min(duration, 300))  # 1~300 프레임 제한
        
        # 더미 키포인트 (중앙 위치 기준, 이미 정규화된 0~1 범위)
        dummy_keypoints = np.full((duration, 133, 2), 0.5, dtype=np.float32)
        # 약간의 노이즈 추가
        dummy_keypoints += np.random.normal(0, 0.05, dummy_keypoints.shape).astype(np.float32)
        dummy_keypoints = np.clip(dummy_keypoints, 0.0, 1.0)
        
        # 더미 스코어 (정규화된 0~1 범위, 낮은 신뢰도)
        dummy_scores = np.full((duration, 133), 0.1, dtype=np.float32)
        
        logger.debug(f"더미 포즈 데이터 생성 (정규화됨): {dummy_keypoints.shape}")
        return dummy_keypoints, dummy_scores
    
    def __len__(self):
        return len(self.valid_segments)
    
    def __getitem__(self, idx):
        """데이터셋 아이템 가져오기 (안정적인 인덱싱)"""
        # 인덱스 범위 확인
        if not (0 <= idx < len(self.valid_segments)):
            idx = idx % len(self.valid_segments)
        
        max_retries = min(20, max(5, len(self.valid_segments) // 20))  # 재시도 횟수 증가
        original_idx = idx
        tried_indices = set()
        
        for attempt in range(max_retries):
            current_idx = idx % len(self.valid_segments)
            
            # 이미 시도한 인덱스는 건너뛰기
            if current_idx in tried_indices:
                idx = (idx + 1) % len(self.valid_segments)
                continue
                
            tried_indices.add(current_idx)
            
            try:
                segment = self.valid_segments[current_idx]
                
                # 포즈 데이터 로드 시도
                pose_data = self._get_pose_data(segment)
                
                if pose_data is None:
                    logger.debug(f"❌ 세그먼트 {current_idx} (Data ID: {segment['data_id']}) 포즈 데이터 없음")
                    idx = (idx + 1) % len(self.valid_segments)
                    continue
                
                segment_keypoints, segment_scores = pose_data
                
                # 유효한 데이터인지 확인
                if segment_keypoints.shape[0] == 0 or segment_scores.shape[0] == 0:
                    logger.debug(f"❌ 세그먼트 {current_idx} 빈 데이터")
                    idx = (idx + 1) % len(self.valid_segments)
                    continue
                
                # 데이터 증강 적용 (훈련 시에만)
                if self.enable_augmentation and self.augmentation is not None:
                    try:
                        segment_keypoints, segment_scores = self.augmentation(segment_keypoints, segment_scores)
                    except Exception as aug_e:
                        logger.debug(f"⚠️ 데이터 증강 실패 (인덱스 {current_idx}): {aug_e}")
                        # 증강 실패해도 원본 데이터 사용
                
                # 포즈 특징 결합
                pose_features = np.concatenate([
                    segment_keypoints,  # [frames, 133, 2] - 정규화됨 (0~1)
                    np.expand_dims(segment_scores, axis=-1)  # [frames, 133, 1] - 정규화됨 (0~1)
                ], axis=-1)  # [frames, 133, 3]
                
                # PyTorch 텐서로 변환
                pose_features = torch.from_numpy(pose_features).float()
                
                # vocab_ids 안전한 변환
                try:
                    if isinstance(segment['vocab_ids'], (list, tuple)):
                        vocab_array = np.array(segment['vocab_ids'][:segment['vocab_len']], dtype=np.int64)
                    else:
                        vocab_array = segment['vocab_ids'][:segment['vocab_len']].astype(np.int64)
                    vocab_ids = torch.from_numpy(vocab_array)
                except Exception as vocab_e:
                    logger.debug(f"⚠️ vocab_ids 변환 실패 (인덱스 {current_idx}): {vocab_e}")
                    vocab_ids = torch.zeros(1, dtype=torch.long)
                
                # 성공적으로 로드된 경우
                return {
                    'pose_features': pose_features,
                    'vocab_ids': vocab_ids,
                    'vocab_len': min(segment['vocab_len'], len(vocab_ids)),
                    'duration': pose_features.shape[0],
                    'segment_info': {
                        'data_id': segment['data_id'],
                        'real_id': segment['real_id'],
                        'start_frame': segment['start_frame'],
                        'end_frame': segment['end_frame']
                    }
                }
                
            except IndexError as e:
                logger.debug(f"❌ 인덱스 오류 (인덱스 {current_idx}): {e}")
                idx = (idx + 1) % len(self.valid_segments)
                continue
            except Exception as e:
                logger.debug(f"❌ 세그먼트 {current_idx} 처리 오류: {e}")
                idx = (idx + 1) % len(self.valid_segments)
                continue
        
        # 모든 시도 실패 시 안전한 더미 샘플 반환 (경고 수준 낮춤)
        if len(tried_indices) >= max_retries:
            logger.debug(f"인덱스 {original_idx} 주변에서 유효한 데이터를 찾지 못함. 더미 샘플 사용")
        
        return self._create_safe_dummy_sample(original_idx)
    
    def _create_safe_dummy_sample(self, idx):
        """안전한 더미 샘플 생성 (정규화된 범위)"""
        logger.debug(f"더미 샘플 생성 (인덱스: {idx})")
        
        # 기본 더미 데이터 (이미 정규화된 0~1 범위)
        dummy_frames = 30
        pose_features = torch.full((dummy_frames, 133, 3), 0.5).float()
        
        # 약간의 랜덤성 추가 (정규화된 범위 유지)
        pose_features += torch.randn_like(pose_features) * 0.1
        pose_features = torch.clamp(pose_features, 0.0, 1.0)
        
        vocab_ids = torch.zeros(1).long()  # 더미 단어 ID
        
        return {
            'pose_features': pose_features,
            'vocab_ids': vocab_ids,
            'vocab_len': 1,
            'duration': dummy_frames,
            'segment_info': {
                'data_id': -1,
                'real_id': -1,
                'start_frame': 0,
                'end_frame': dummy_frames
            }
        }


def collate_fn(batch):
    """배치 처리 함수 - 가변 길이 시퀀스 패딩"""
    batch_size = len(batch)
    
    # 최대 길이 계산
    max_frames = max(item['duration'] for item in batch)
    max_vocab_len = max(item['vocab_len'] for item in batch)
    
    # 패딩된 배치 텐서 초기화
    batch_pose_features = torch.zeros(batch_size, max_frames, 133, 3)
    batch_vocab_ids = torch.zeros(batch_size, max_vocab_len, dtype=torch.long)
    batch_frame_masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    batch_vocab_masks = torch.zeros(batch_size, max_vocab_len, dtype=torch.bool)
    
    # 메타데이터
    batch_vocab_lens = []
    batch_durations = []
    batch_segment_infos = []
    
    for i, item in enumerate(batch):
        frames = item['duration']
        vocab_len = item['vocab_len']
        
        # 포즈 데이터 복사
        batch_pose_features[i, :frames] = item['pose_features']
        batch_frame_masks[i, :frames] = True
        
        # Vocabulary 데이터 복사
        batch_vocab_ids[i, :vocab_len] = item['vocab_ids']
        batch_vocab_masks[i, :vocab_len] = True
        
        # 메타데이터
        batch_vocab_lens.append(vocab_len)
        batch_durations.append(frames)
        batch_segment_infos.append(item['segment_info'])
    
    return {
        'pose_features': batch_pose_features,  # [batch, max_frames, 133, 3]
        'vocab_ids': batch_vocab_ids,  # [batch, max_vocab_len]
        'frame_masks': batch_frame_masks,  # [batch, max_frames]
        'vocab_masks': batch_vocab_masks,  # [batch, max_vocab_len]
        'vocab_lens': torch.tensor(batch_vocab_lens),
        'durations': torch.tensor(batch_durations),
        'segment_infos': batch_segment_infos
    }

def create_dataloader(dataset_or_path,
                     batch_size: int = 8,
                     shuffle: bool = True,
                     num_workers: int = 2,
                     annotation_path: str = "./data/sign_language_dataset_only_sen_lzf.h5",
                     pose_data_dir: str = "./data",
                     enable_augmentation: bool = False,
                     augmentation_config: Dict = None,
                     **kwargs):
    """통합 데이터 로더 생성 - Dataset 객체 또는 경로 지원"""
    
    # Dataset 또는 Subset 객체인지 확인
    if hasattr(dataset_or_path, '__getitem__') and hasattr(dataset_or_path, '__len__'):
        # 이미 Dataset 또는 Subset 객체
        dataset = dataset_or_path
    else:
        # 경로가 주어진 경우 새 Dataset 생성
        dataset = UnifiedSignLanguageDataset(
            annotation_path=annotation_path,
            pose_data_dir=pose_data_dir,
            enable_augmentation=enable_augmentation,
            augmentation_config=augmentation_config,
            **kwargs
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # XPU 환경 고려
    )
    
    return dataloader, dataset

# 테스트 코드
if __name__ == "__main__":
    print("🚀 통합 포즈 데이터 로더 테스트 (증강 포함)")
    print("=" * 60)
    
    try:
        # 기본 데이터셋 생성 (증강 비활성화)
        print("\n1️⃣ 기본 데이터셋 테스트 (증강 없음)")
        dataset_basic = UnifiedSignLanguageDataset(
            annotation_path="./data/sign_language_dataset_only_sen_lzf.h5",
            pose_data_dir="./data",
            sequence_length=200,
            min_segment_length=20,
            max_segment_length=300,
            enable_augmentation=False
        )
        
        print(f"✅ 기본 데이터셋 로드 완료:")
        print(f"   - 총 세그먼트: {len(dataset_basic)} 개")
        print(f"   - Vocabulary 크기: {dataset_basic.vocab_size}")
        
        # 증강 데이터셋 생성
        print("\n2️⃣ 증강 데이터셋 테스트 (모든 증강 활성화)")
        augmentation_config = {
            'enable_horizontal_flip': True,
            'enable_rotation': True, 
            'enable_scaling': True,
            'enable_noise': True,
            'horizontal_flip_prob': 0.7,
            'rotation_range': 15.0,
            'scaling_range': (0.85, 1.15),
            'noise_std': 0.02
        }
        
        dataset_aug = UnifiedSignLanguageDataset(
            annotation_path="./data/sign_language_dataset_only_sen_lzf.h5",
            pose_data_dir="./data",
            sequence_length=200,
            min_segment_length=20,
            max_segment_length=300,
            enable_augmentation=True,
            augmentation_config=augmentation_config
        )
        
        # 동일한 샘플로 증강 효과 비교
        print(f"\n3️⃣ 증강 효과 비교 테스트")
        
        # 기본 샘플
        basic_sample = dataset_basic[0]
        basic_pose = basic_sample['pose_features'][:, :, :2]  # [frames, 133, 2]
        
        print(f"   📊 기본 샘플 통계:")
        print(f"     - 프레임 수: {basic_sample['duration']}")
        print(f"     - 키포인트 X 범위: [{basic_pose[:, :, 0].min():.3f}, {basic_pose[:, :, 0].max():.3f}]")
        print(f"     - 키포인트 Y 범위: [{basic_pose[:, :, 1].min():.3f}, {basic_pose[:, :, 1].max():.3f}]")
        
        # 증강 샘플들 (같은 인덱스에서 여러번 추출)
        aug_samples = []
        for i in range(3):
            aug_sample = dataset_aug[0]  # 동일 인덱스에서 증강
            aug_samples.append(aug_sample)
        
        print(f"\n   🎨 증강된 샘플들 비교:")
        for i, aug_sample in enumerate(aug_samples):
            aug_pose = aug_sample['pose_features'][:, :, :2]
            print(f"     증강 {i+1}: X[{aug_pose[:, :, 0].min():.3f}, {aug_pose[:, :, 0].max():.3f}], "
                  f"Y[{aug_pose[:, :, 1].min():.3f}, {aug_pose[:, :, 1].max():.3f}]")
        
        # 배치 테스트
        print(f"\n4️⃣ 배치 처리 테스트")
        dataloader_basic = DataLoader(dataset_basic, batch_size=2, shuffle=False, collate_fn=collate_fn)
        dataloader_aug = DataLoader(dataset_aug, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        batch_basic = next(iter(dataloader_basic))
        batch_aug = next(iter(dataloader_aug))
        
        print(f"   기본 배치: {batch_basic['pose_features'].shape}")
        print(f"   증강 배치: {batch_aug['pose_features'].shape}")
        print(f"   평균 프레임 수: 기본 {batch_basic['durations'].float().mean():.1f}, "
              f"증강 {batch_aug['durations'].float().mean():.1f}")
        
        print(f"\n✅ 모든 테스트 성공!")
        
    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()
