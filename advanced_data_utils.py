"""
고급 데이터 분할 및 관리 유틸리티
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import logging

from unified_pose_dataloader import UnifiedSignLanguageDataset, collate_fn
from advanced_config import DataSplitConfig, RandomSeedConfig

logger = logging.getLogger(__name__)

class StratifiedDataSplitter:
    """단어별 균등 분할을 위한 클래스"""
    
    def __init__(self, config: DataSplitConfig, random_seed: int = 42):
        self.config = config
        self.random_seed = random_seed
        
    def split_dataset_stratified(self, dataset: UnifiedSignLanguageDataset) -> Tuple[List[int], List[int], List[int]]:
        """단어별로 균등하게 데이터 분할"""
        
        # 단어별 인덱스 수집
        word_to_indices = defaultdict(list)
        
        logger.info("📊 단어별 데이터 분포 분석 중...")
        for idx, segment in enumerate(dataset.valid_segments):
            for word_id in segment['vocab_ids']:
                if word_id > 0:  # 패딩이 아닌 실제 단어만
                    word_to_indices[word_id].append(idx)
        
        # 단어별 분포 출력
        word_counts = {word_id: len(indices) for word_id, indices in word_to_indices.items()}
        logger.info(f"총 {len(word_counts)}개 단어의 데이터 분포:")
        
        # 상위 10개 단어 출력
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for word_id, count in top_words:
            # vocab 속성이 있는지 확인하고 안전하게 접근
            if hasattr(dataset, 'vocab') and dataset.vocab:
                word_name = dataset.vocab.get(word_id, f"ID_{word_id}")
            else:
                word_name = f"ID_{word_id}"
            logger.info(f"  {word_name}: {count}개")
        
        # 분할 실행
        train_indices, val_indices, test_indices = [], [], []
        insufficient_data_words = []
        
        for word_id, indices in word_to_indices.items():
            # vocab 속성이 있는지 확인하고 안전하게 접근
            if hasattr(dataset, 'vocab') and dataset.vocab:
                word_name = dataset.vocab.get(word_id, f"ID_{word_id}")
            else:
                word_name = f"ID_{word_id}"
            
            # 최소 샘플 수 확인
            min_total = (self.config.min_samples_per_word_train + 
                        self.config.min_samples_per_word_val + 
                        self.config.min_samples_per_word_test)
            
            if len(indices) < min_total:
                insufficient_data_words.append((word_name, len(indices)))
                # 데이터가 부족한 단어는 훈련 세트에만 포함
                train_indices.extend(indices)
                continue
            
            # Stratified split
            np.random.seed(self.random_seed)
            shuffled_indices = np.array(indices)
            np.random.shuffle(shuffled_indices)
            
            # 비율에 따른 분할
            n_samples = len(shuffled_indices)
            n_train = max(int(n_samples * self.config.train_ratio), 
                         self.config.min_samples_per_word_train)
            n_val = max(int(n_samples * self.config.val_ratio), 
                       self.config.min_samples_per_word_val)
            n_test = max(int(n_samples * self.config.test_ratio), 
                        self.config.min_samples_per_word_test)
            
            # 오버플로우 방지
            if n_train + n_val + n_test > n_samples:
                n_train = n_samples - n_val - n_test
                if n_train < self.config.min_samples_per_word_train:
                    n_val = max(1, n_samples - n_train - n_test)
                    n_test = max(1, n_samples - n_train - n_val)
            
            # 실제 분할
            train_indices.extend(shuffled_indices[:n_train].tolist())
            val_indices.extend(shuffled_indices[n_train:n_train+n_val].tolist())
            test_indices.extend(shuffled_indices[n_train+n_val:n_train+n_val+n_test].tolist())
        
        # 경고 출력
        if insufficient_data_words:
            logger.warning(f"⚠️ 데이터가 부족한 {len(insufficient_data_words)}개 단어:")
            for word_name, count in insufficient_data_words[:5]:
                logger.warning(f"  {word_name}: {count}개 (최소 {min_total}개 필요)")
        
        # 최종 분할 결과
        logger.info("✅ 데이터 분할 완료:")
        logger.info(f"  훈련: {len(train_indices)}개 ({len(train_indices)/len(dataset)*100:.1f}%)")
        logger.info(f"  검증: {len(val_indices)}개 ({len(val_indices)/len(dataset)*100:.1f}%)")
        logger.info(f"  테스트: {len(test_indices)}개 ({len(test_indices)/len(dataset)*100:.1f}%)")
        
        return train_indices, val_indices, test_indices

    def create_dataloaders(self, 
                          dataset: UnifiedSignLanguageDataset,
                          batch_size: int = 32,
                          enable_train_augmentation: bool = False,
                          augmentation_config: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터로더 생성"""
        
        # 데이터 분할
        train_indices, val_indices, test_indices = self.split_dataset_stratified(dataset)
        
        # 증강 적용 여부에 따라 다른 데이터셋 생성
        if enable_train_augmentation:
            # 훈련용: 증강 활성화
            train_dataset_aug = UnifiedSignLanguageDataset(
                annotation_path=dataset.annotation_path,
                pose_data_dir=dataset.pose_data_dir,
                sequence_length=dataset.sequence_length,
                min_segment_length=dataset.min_segment_length,
                max_segment_length=dataset.max_segment_length,
                enable_augmentation=True,
                augmentation_config=augmentation_config
            )
            train_dataset = Subset(train_dataset_aug, train_indices)
        else:
            train_dataset = Subset(dataset, train_indices)
        
        # 검증/테스트용: 증강 비활성화
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # 데이터로더 생성
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
        return train_dataloader, val_dataloader, test_dataloader

class EarlyStopping:
    """개선된 얼리스탑 클래스"""
    
    def __init__(self, config):
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.monitor = config.monitor
        self.mode = config.mode
        self.restore_best_weights = config.restore_best_weights
        
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
    def __call__(self, current_score: float, epoch: int, model_state_dict: Dict) -> bool:
        """얼리스탑 체크"""
        
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model_state_dict.items()}
            logger.info(f"🎉 새로운 최고 성능! {self.monitor}: {current_score:.6f}")
        else:
            self.wait += 1
            logger.info(f"⏳ 개선 없음 ({self.wait}/{self.patience}) - 현재: {current_score:.6f}, 최고: {self.best_score:.6f}")
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(f"⏹️ 얼리스탑 발동! (에포크 {epoch})")
            return True
        
        return False
    
    def get_best_weights(self):
        """최고 성능 가중치 반환"""
        return self.best_weights