#!/usr/bin/env python3
"""
간단한 다단계 학습 트레이너 (테스트용)
"""
import os
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np

from sign_language_model import SequenceToSequenceSignModel
from sign_language_trainer import SignLanguageTrainer
from advanced_config import AdvancedTrainingConfig, TrainingStageConfig
from advanced_data_utils import StratifiedDataSplitter
from unified_pose_dataloader import UnifiedSignLanguageDataset

logger = logging.getLogger(__name__)

class SimpleEarlyStopping:
    """간단한 얼리스탑 클래스"""
    
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
    def __call__(self, current_score: float, epoch: int, model_state_dict: Dict) -> bool:
        """얼리스탑 체크"""
        
        if current_score < (self.best_score - self.min_delta):
            self.best_score = current_score
            self.wait = 0
            self.best_weights = {k: v.clone() for k, v in model_state_dict.items()}
            logger.info(f"🎉 새로운 최고 성능! Val Loss: {current_score:.6f}")
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

class SimpleAdvancedTrainer:
    """간단한 다단계 학습 트레이너"""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        
        # 랜덤시드 고정
        self.config.random_seed.fix_all_seeds()
        logger.info(f"🎲 랜덤시드 고정: {self.config.random_seed.seed}")
        
        # 디렉토리 생성
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 현재 최고 성능
        self.best_overall_performance = {
            'stage': None,
            'epoch': None,
            'val_loss': float('inf'),
            'val_accuracy': 0.0
        }
        
    def setup_device(self):
        """디바이스 설정"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"🔥 CUDA 디바이스 사용: {torch.cuda.get_device_name()}")
            else:
                try:
                    import intel_extension_for_pytorch as ipex
                    device = torch.device("xpu")
                    logger.info("⚡ XPU 디바이스 사용")
                except:
                    device = torch.device("cpu")
                    logger.info("💻 CPU 디바이스 사용")
        else:
            device = torch.device(self.config.device)
            logger.info(f"🎯 지정된 디바이스 사용: {device}")
        
        return device
    
    def load_and_prepare_data(self):
        """데이터 로드 및 분할 준비"""
        logger.info("📊 데이터 로드 및 분할 준비...")
        
        dataset = UnifiedSignLanguageDataset(
            annotation_path=self.config.annotation_path,
            pose_data_dir=self.config.pose_data_dir,
            sequence_length=200,
            min_segment_length=20,
            max_segment_length=300,
            enable_augmentation=False
        )
        
        logger.info(f"✅ 기본 데이터셋 로드 완료: {len(dataset)}개 세그먼트")
        
        # 데이터 분할기 생성
        splitter = StratifiedDataSplitter(
            config=self.config.data_split,
            random_seed=self.config.random_seed.seed
        )
        
        return dataset, splitter
    
    def create_model(self, vocab_size: int) -> SequenceToSequenceSignModel:
        """모델 생성"""
        model = SequenceToSequenceSignModel(
            vocab_size=vocab_size,
            embed_dim=256,  # 작게 시작
            num_encoder_layers=4,  # 작게 시작
            num_decoder_layers=2,  # 작게 시작
            num_heads=8,
            dim_feedforward=512,  # 작게 시작
            max_seq_len=200,
            dropout=0.1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"📊 모델 생성: {total_params:,} 파라미터")
        
        return model
    
    def train_single_stage(self, 
                          stage_idx: int, 
                          stage_config: TrainingStageConfig, 
                          model: SequenceToSequenceSignModel,
                          base_dataset: UnifiedSignLanguageDataset,
                          splitter: StratifiedDataSplitter,
                          device: torch.device) -> Dict:
        """단일 단계 학습"""
        
        logger.info("="*60)
        logger.info(f"🚀 Stage {stage_idx+1}: {stage_config.name}")
        logger.info(f"📝 {stage_config.description}")
        logger.info(f"⚙️ 에포크: {stage_config.num_epochs}, 배치: {stage_config.batch_size}")
        logger.info(f"⚙️ 학습률: {stage_config.learning_rate}, 드롭아웃: {stage_config.dropout_rate}")
        logger.info("="*60)
        
        # 증강 설정
        augmentation_config = None
        if stage_config.enable_augmentation:
            augmentation_config = {
                'enable_horizontal_flip': True,
                'enable_rotation': True,
                'enable_scaling': True,
                'enable_noise': True,
                'horizontal_flip_prob': 0.5 * stage_config.augmentation_strength,
                'rotation_range': 10.0 * stage_config.augmentation_strength,
                'scaling_range': (1.0 - 0.05 * stage_config.augmentation_strength, 
                                 1.0 + 0.05 * stage_config.augmentation_strength),
                'noise_std': 0.005 * stage_config.augmentation_strength
            }
        
        # 데이터로더 생성
        train_dataloader, val_dataloader, test_dataloader = splitter.create_dataloaders(
            dataset=base_dataset,
            batch_size=stage_config.batch_size,
            enable_train_augmentation=stage_config.enable_augmentation,
            augmentation_config=augmentation_config
        )
        
        # 옵티마이저 생성
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=stage_config.learning_rate,
            weight_decay=stage_config.weight_decay
        )
        
        # 스케줄러 생성
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.7
        )
        
        # 얼리스탑 설정
        early_stopping = SimpleEarlyStopping(patience=8, min_delta=1e-4)
        
        # 학습 루프
        stage_start_time = time.time()
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_epoch = -1
        
        for epoch in range(stage_config.num_epochs):
            epoch_start_time = time.time()
            
            # 훈련
            model.train()
            total_train_loss = 0
            train_batches = 0
            
            train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch:2d} 훈련")
            for batch in train_pbar:
                optimizer.zero_grad()
                
                # 데이터를 디바이스로 이동
                pose_features = batch['pose_features'].to(device)
                vocab_ids = batch['vocab_ids'].to(device)
                frame_masks = batch['frame_masks'].to(device)
                vocab_masks = batch['vocab_masks'].to(device)
                
                # 순전파
                outputs = model(
                    pose_features=pose_features,
                    vocab_ids=vocab_ids,
                    frame_masks=frame_masks,
                    vocab_masks=vocab_masks
                )
                
                # 손실 계산 (간단한 버전)
                if isinstance(outputs, dict) and 'vocab_logits' in outputs:
                    vocab_logits = outputs['vocab_logits']  # [batch, vocab_len, vocab_size]
                    
                    # 타겟 준비 (다음 단어 예측)
                    targets = vocab_ids[:, 1:].contiguous()  # 다음 단어들
                    logits = vocab_logits[:, :-1].contiguous()  # 마지막 예측 제외
                    
                    # 마스크 준비
                    target_mask = vocab_masks[:, 1:].contiguous()
                    
                    # 손실 계산
                    vocab_loss = nn.CrossEntropyLoss(ignore_index=0)(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                    
                    loss = vocab_loss
                else:
                    # 백업: 더미 손실
                    loss = torch.tensor(0.0, requires_grad=True, device=device)
                
                # 역전파
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                train_batches += 1
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_train_loss/train_batches:.4f}'
                })
            
            avg_train_loss = total_train_loss / max(train_batches, 1)
            
            # 검증
            model.eval()
            total_val_loss = 0
            val_batches = 0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch:2d} 검증")
                for batch in val_pbar:
                    pose_features = batch['pose_features'].to(device)
                    vocab_ids = batch['vocab_ids'].to(device)
                    frame_masks = batch['frame_masks'].to(device)
                    vocab_masks = batch['vocab_masks'].to(device)
                    
                    outputs = model(
                        pose_features=pose_features,
                        vocab_ids=vocab_ids,
                        frame_masks=frame_masks,
                        vocab_masks=vocab_masks
                    )
                    
                    # 검증 손실 계산 (훈련과 동일)
                    if isinstance(outputs, dict) and 'vocab_logits' in outputs:
                        vocab_logits = outputs['vocab_logits']
                        targets = vocab_ids[:, 1:].contiguous()
                        logits = vocab_logits[:, :-1].contiguous()
                        
                        vocab_loss = nn.CrossEntropyLoss(ignore_index=0)(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1)
                        )
                        
                        loss = vocab_loss
                        
                        # 정확도 계산 (간단한 버전)
                        predictions = torch.argmax(logits, dim=-1)
                        mask = targets != 0
                        correct_predictions += (predictions == targets)[mask].sum().item()
                        total_predictions += mask.sum().item()
                    else:
                        loss = torch.tensor(0.0, device=device)
                    
                    total_val_loss += loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{total_val_loss/val_batches:.4f}'
                    })
            
            avg_val_loss = total_val_loss / max(val_batches, 1)
            val_accuracy = correct_predictions / max(total_predictions, 1)
            
            # 성능 기록
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_accuracy = val_accuracy
                best_epoch = epoch
            
            # 스케줄러 업데이트
            scheduler.step(avg_val_loss)
            
            # 로깅
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch:3d}/{stage_config.num_epochs-1} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            logger.info(f"  Val Acc: {val_accuracy:.3f} | Best Val Loss: {best_val_loss:.4f}")
            
            # 얼리스탑 체크
            if early_stopping(avg_val_loss, epoch, model.state_dict()):
                logger.info(f"⏹️ 얼리스탑 발동 - Stage {stage_idx+1} 종료")
                
                # 최고 성능 가중치 복원
                if early_stopping.best_weights:
                    model.load_state_dict(early_stopping.best_weights)
                    logger.info("✅ 최고 성능 가중치 복원 완료")
                break
        
        # 단계 결과
        stage_results = {
            'stage_name': stage_config.name,
            'description': stage_config.description,
            'epochs_trained': epoch + 1,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_accuracy,
            'training_time': time.time() - stage_start_time
        }
        
        # 전체 최고 성능 업데이트
        if best_val_loss < self.best_overall_performance['val_loss']:
            self.best_overall_performance.update({
                'stage': stage_config.name,
                'epoch': best_epoch,
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy
            })
            
            # 최고 성능 모델 저장
            best_model_path = self.checkpoint_dir / f"best_model_stage_{stage_idx+1}.pt"
            torch.save({
                'stage': stage_config.name,
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': best_val_accuracy
            }, best_model_path)
            logger.info(f"💾 최고 성능 모델 저장: {best_model_path}")
        
        logger.info(f"✅ Stage {stage_idx+1} 완료!")
        logger.info(f"  최고 성능: Val Loss {best_val_loss:.4f}, Val Acc {best_val_accuracy:.3f}")
        logger.info(f"  학습 시간: {stage_results['training_time']:.1f}초")
        
        return stage_results
    
    def train_multi_stage(self) -> Dict:
        """다단계 학습 실행"""
        logger.info("🚀 간단한 다단계 학습 시작!")
        
        # 디바이스 설정
        device = self.setup_device()
        
        # 데이터 준비
        base_dataset, splitter = self.load_and_prepare_data()
        
        # 모델 생성
        model = self.create_model(base_dataset.vocab_size)
        model.to(device)
        
        # 각 단계별 학습
        all_stage_results = []
        
        for stage_idx, stage_config in enumerate(self.config.multi_stage.stages[:2]):  # 처음 2단계만
            stage_results = self.train_single_stage(
                stage_idx, stage_config, model, base_dataset, splitter, device
            )
            all_stage_results.append(stage_results)
            
            # 너무 성능이 안 좋으면 중단
            if stage_results['best_val_loss'] > 10.0:
                logger.warning("⚠️ 학습 성능이 너무 나쁨. 조기 종료")
                break
        
        # 결과 요약
        logger.info("\n" + "="*60)
        logger.info("🎉 다단계 학습 완료!")
        logger.info("="*60)
        
        for i, stage in enumerate(all_stage_results):
            logger.info(f"Stage {i+1} ({stage['stage_name']}):")
            logger.info(f"  Val Loss: {stage['best_val_loss']:.4f}")
            logger.info(f"  Val Acc: {stage['best_val_accuracy']:.3f}")
            logger.info(f"  시간: {stage['training_time']:.1f}초")
        
        logger.info(f"\n🏆 최고 성능: {self.best_overall_performance['stage']}")
        logger.info(f"  Val Loss: {self.best_overall_performance['val_loss']:.4f}")
        logger.info(f"  Val Acc: {self.best_overall_performance['val_accuracy']:.3f}")
        
        return {
            'stages': all_stage_results,
            'best_performance': self.best_overall_performance
        }

if __name__ == "__main__":
    # 간단한 테스트
    logging.basicConfig(level=logging.INFO)
    
    config = AdvancedTrainingConfig()
    # 빠른 테스트를 위한 설정 조정
    for stage in config.multi_stage.stages:
        stage.num_epochs = 3
        stage.batch_size = 8
    
    trainer = SimpleAdvancedTrainer(config)
    results = trainer.train_multi_stage()
    
    print("✅ 간단한 다단계 학습 테스트 완료!")
