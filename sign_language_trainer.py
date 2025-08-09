#!/usr/bin/env python3
"""
수화 인식 모델 트레이너
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

from unified_pose_dataloader import create_dataloader
from device_utils import DeviceManager
from sign_language_model import SequenceToSequenceSignModel, RealtimeDecoder

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageTrainer:
    """수화 인식 모델 트레이너"""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader,
                 val_dataloader,
                 device: str = "auto",
                 checkpoint_dir: str = "./checkpoints",
                 log_dir: str = "./logs",
                 vocab_words: List[str] = None):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.vocab_words = vocab_words
        
        # 종료 신호 플래그
        self.shutdown_requested = False
        self.current_epoch = 0
        self.current_step = 0
        
        # 장치 설정 (개선된 방식)
        self.device = DeviceManager.detect_best_device(device)
        device_info = DeviceManager.get_device_info(self.device)
        logger.info(f"🚀 디바이스 설정: {device_info.get('name', self.device)}")
        
        # 디바이스별 최적화
        DeviceManager.optimize_for_device(self.device)
        
        self.vocab_words = vocab_words or []
        
        # 디렉토리 생성
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir / f"train_{int(time.time())}")
        
        # 모델을 GPU로 이동
        self.model.to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
            betas=(0.9, 0.98)
        )
        
        # 스케줄러
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # 학습 통계
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stopping_patience = 15
        
        # 메트릭
        self.train_metrics = {
            'word_accuracy': [],
            'boundary_accuracy': [],
            'confidence_mae': []
        }
        self.val_metrics = {
            'word_accuracy': [],
            'boundary_accuracy': [],
            'confidence_mae': []
        }
        
        logger.info(f"🔧 트레이너 초기화 완료:")
        logger.info(f"   - 디바이스: {self.device}")
        logger.info(f"   - 모델 파라미터: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   - 훈련 배치: {len(self.train_dataloader)}")
        logger.info(f"   - 검증 배치: {len(self.val_dataloader)}")
    
    def request_shutdown(self):
        """종료 요청"""
        logger.info("⚠️ 안전한 종료가 요청되었습니다.")
        self.shutdown_requested = True
    
    def setup_training(self, 
                      learning_rate: float = 1e-4, 
                      weight_decay: float = 1e-2,
                      warmup_steps: int = 500,
                      gradient_clip_val: float = 1.0,
                      word_loss_weight: float = 1.0,
                      boundary_loss_weight: float = 0.5,
                      confidence_loss_weight: float = 0.3):
        """학습 설정 업데이트"""
        logger.info(f"⚙️ 학습 파라미터 설정:")
        logger.info(f"   - Learning Rate: {learning_rate}")
        logger.info(f"   - Weight Decay: {weight_decay}")
        logger.info(f"   - Warmup Steps: {warmup_steps}")
        logger.info(f"   - Gradient Clip: {gradient_clip_val}")
        
        # 학습 파라미터 저장
        self.gradient_clip_val = gradient_clip_val
        self.word_loss_weight = word_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.confidence_loss_weight = confidence_loss_weight
        
        # 옵티마이저 재설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )
        
        # 스케줄러 재설정
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    
    def calculate_metrics(self, outputs: Dict, targets: Dict, vocab_masks: torch.Tensor) -> Dict:
        """메트릭 계산"""
        metrics = {}
        
        # 단어 정확도 계산
        if 'word_logits' in outputs and 'vocab_ids' in targets:
            word_logits = outputs['word_logits']  # [batch, vocab_len, vocab_size]
            target_ids = targets['vocab_ids']  # [batch, vocab_len]
            
            # 예측값 계산
            predicted_ids = torch.argmax(word_logits, dim=-1)
            
            # 마스크된 위치에서만 정확도 계산
            correct_predictions = (predicted_ids == target_ids) & vocab_masks
            word_accuracy = correct_predictions.sum().float() / vocab_masks.sum().float()
            metrics['word_accuracy'] = word_accuracy.item()
        
        # 경계 정확도 계산
        if 'boundary_logits' in outputs and 'boundary_labels' in targets:
            boundary_logits = outputs['boundary_logits']  # [batch, vocab_len, 3]
            boundary_labels = targets['boundary_labels']  # [batch, vocab_len]
            
            predicted_boundaries = torch.argmax(boundary_logits, dim=-1)
            boundary_correct = (predicted_boundaries == boundary_labels) & vocab_masks
            boundary_accuracy = boundary_correct.sum().float() / vocab_masks.sum().float()
            metrics['boundary_accuracy'] = boundary_accuracy.item()
        
        # 신뢰도 MAE 계산
        if 'confidence_scores' in outputs and 'confidence_targets' in targets:
            confidence_scores = outputs['confidence_scores']  # [batch, vocab_len, 1]
            confidence_targets = targets['confidence_targets']  # [batch, vocab_len, 1]
            
            # 유효한 위치에서만 MAE 계산
            masked_diff = torch.abs(confidence_scores - confidence_targets)
            confidence_mae = masked_diff[vocab_masks.unsqueeze(-1)].mean()
            metrics['confidence_mae'] = confidence_mae.item()
        
        return metrics
        
    def create_boundary_labels(self, vocab_ids, vocab_masks):
        """경계 라벨 생성 (단순 버전)"""
        batch_size, vocab_len = vocab_ids.shape
        boundary_labels = torch.zeros(batch_size, vocab_len, dtype=torch.long, device=vocab_ids.device)
        
        for i in range(batch_size):
            mask = vocab_masks[i]
            valid_len = mask.sum().item()
            
            if valid_len > 0:
                boundary_labels[i, 0] = 0  # START
                boundary_labels[i, 1:valid_len-1] = 1  # CONTINUE
                if valid_len > 1:
                    boundary_labels[i, valid_len-1] = 2  # END
        
        return boundary_labels
    
    def compute_metrics(self, outputs, targets, masks):
        """메트릭 계산 (통합 버전)"""
        vocab_masks = masks.get('vocab_masks', masks)  # 호환성을 위해
        return self.calculate_metrics(outputs, targets, vocab_masks)
        word_correct = (word_preds == vocab_ids) & vocab_masks
        word_accuracy = word_correct.sum().float() / vocab_masks.sum().float()
        metrics['word_accuracy'] = word_accuracy.item()
        
        # 경계 정확도 (있는 경우)
        if 'boundary_logits' in outputs and 'boundary_labels' in targets:
            boundary_logits = outputs['boundary_logits']
            boundary_labels = targets['boundary_labels']
            
            boundary_preds = torch.argmax(boundary_logits, dim=-1)
            boundary_correct = (boundary_preds == boundary_labels) & vocab_masks
            boundary_accuracy = boundary_correct.sum().float() / vocab_masks.sum().float()
            metrics['boundary_accuracy'] = boundary_accuracy.item()
        
        # 신뢰도 MAE (있는 경우)
        if 'confidence_scores' in outputs and 'confidence_targets' in targets:
            confidence_scores = outputs['confidence_scores']
            confidence_targets = targets['confidence_targets']
            
            confidence_mae = torch.abs(confidence_scores - confidence_targets)[vocab_masks].mean()
            metrics['confidence_mae'] = confidence_mae.item()
        
        return metrics
    
    def train_epoch(self, epoch: int, save_every_n_steps: int = 500, log_every_n_steps: int = 100):
        """한 에포크 훈련 (종료 신호 처리 포함)"""
        self.model.train()
        total_loss = 0
        total_metrics = {key: 0 for key in self.train_metrics.keys()}
        
        pbar = tqdm(self.train_dataloader, desc=f"훈련 Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 종료 요청 확인
            if self.shutdown_requested:
                logger.info(f"⚠️ 종료 요청 감지 - 현재 배치 완료 후 안전 종료 (배치 {batch_idx}/{len(self.train_dataloader)})")
                break
            
            self.current_step += 1
            
            try:
                # 데이터를 GPU로 이동
                pose_features = batch['pose_features'].to(self.device)
                vocab_ids = batch['vocab_ids'].to(self.device)
                frame_masks = batch['frame_masks'].to(self.device)
                vocab_masks = batch['vocab_masks'].to(self.device)
                
                # 경계 라벨 생성
                boundary_labels = self.create_boundary_labels(vocab_ids, vocab_masks)
                
                # 포워드 패스
                outputs = self.model(
                    pose_features=pose_features,
                    vocab_ids=vocab_ids,
                    frame_masks=frame_masks,
                    vocab_masks=vocab_masks
                )
                
                # 타겟 준비
                targets = {
                    'vocab_ids': vocab_ids,
                    'boundary_labels': boundary_labels,
                    # 'confidence_targets': None  # 실제 신뢰도 타겟이 있다면 추가
                }
                
                # 손실 계산
                losses = self.model.compute_loss(outputs, targets, vocab_masks)
                loss = losses['total_loss']
                
                # 백프롭
                self.optimizer.zero_grad()
                loss.backward()
                
                # 그라디언트 클리핑
                if hasattr(self, 'gradient_clip_val'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                
                # 통계 업데이트
                total_loss += loss.item()
                
                # 메트릭 계산
                metrics = self.compute_metrics(outputs, targets, {'vocab_masks': vocab_masks})
                for key, value in metrics.items():
                    if key in total_metrics:
                        total_metrics[key] += value
                
                # 진행률 업데이트
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Word_Acc': f'{total_metrics["word_accuracy"]/(batch_idx+1):.3f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # 주기적 로깅
                if self.current_step % log_every_n_steps == 0:
                    self.writer.add_scalar('Train/Loss_Step', loss.item(), self.current_step)
                    self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.current_step)
                
                # 주기적 체크포인트 저장 (선택적으로 스텝 단위 저장)
                # 에포크 단위 저장을 우선하므로 스텝 단위 저장은 비활성화
                # if self.current_step % save_every_n_steps == 0:
                #     logger.info(f"💾 중간 체크포인트 저장 (스텝 {self.current_step})")
                #     self.save_checkpoint(epoch, f"step_{self.current_step}")
                
            except Exception as e:
                logger.error(f"❌ 배치 처리 중 오류: {e}")
                continue  # 오류 발생 시 다음 배치로 진행
        
        # 에포크 평균 계산
        if len(self.train_dataloader) > 0:
            avg_loss = total_loss / len(self.train_dataloader)
            avg_metrics = {key: value / len(self.train_dataloader) 
                          for key, value in total_metrics.items()}
        else:
            avg_loss = 0
            avg_metrics = {key: 0 for key in total_metrics.keys()}
        
        # 통계 저장
        self.train_losses.append(avg_loss)
        for key, value in avg_metrics.items():
            if key in self.train_metrics:
                self.train_metrics[key].append(value)
        
        return avg_loss, avg_metrics
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0
        total_metrics = {key: 0 for key in self.val_metrics.keys()}
        
        pbar = tqdm(self.val_dataloader, desc=f"검증 Epoch {epoch}")
        
        for batch in pbar:
            # 데이터를 GPU로 이동
            pose_features = batch['pose_features'].to(self.device)
            vocab_ids = batch['vocab_ids'].to(self.device)
            frame_masks = batch['frame_masks'].to(self.device)
            vocab_masks = batch['vocab_masks'].to(self.device)
            
            # 경계 라벨 생성
            boundary_labels = self.create_boundary_labels(vocab_ids, vocab_masks)
            
            # 검증 시에도 훈련 모드와 동일한 출력을 위해 training 플래그 임시 설정
            self.model.train()
            outputs = self.model(
                pose_features=pose_features,
                vocab_ids=vocab_ids,
                frame_masks=frame_masks,
                vocab_masks=vocab_masks
            )
            self.model.eval()
            
            # 타겟 준비
            targets = {
                'vocab_ids': vocab_ids,
                'boundary_labels': boundary_labels,
            }
            
            # 손실 계산
            losses = self.model.compute_loss(outputs, targets, vocab_masks)
            loss = losses['total_loss']
            
            # 통계 업데이트
            total_loss += loss.item()
            
            # 메트릭 계산
            masks = {'vocab_masks': vocab_masks}
            batch_metrics = self.compute_metrics(outputs, targets, masks)
            for key, value in batch_metrics.items():
                if key in total_metrics:
                    total_metrics[key] += value
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'Val Loss': f"{loss.item():.4f}",
                'Word Acc': f"{batch_metrics.get('word_accuracy', 0):.3f}"
            })
        
        # 에포크 평균 계산
        avg_loss = total_loss / len(self.val_dataloader)
        avg_metrics = {key: value / len(self.val_dataloader) 
                      for key, value in total_metrics.items()}
        
        # 통계 저장
        self.val_losses.append(avg_loss)
        for key, value in avg_metrics.items():
            self.val_metrics[key].append(value)
        
        # TensorBoard 로깅
        self.writer.add_scalar('Val/Epoch_Loss', avg_loss, epoch)
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'Val/Epoch_{key}', value, epoch)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, suffix: str = None):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'vocab_words': self.vocab_words
        }
        
        # 일반 체크포인트 (suffix가 있으면 추가)
        if suffix:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_{suffix}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 체크포인트
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 최고 성능 모델 저장: {best_path}")
        
        # 최신 체크포인트 (suffix가 없을 때만)
        if not suffix:
            latest_path = self.checkpoint_dir / "latest_model.pt"
            torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics = checkpoint['train_metrics']
        self.val_metrics = checkpoint['val_metrics']
        self.best_val_loss = checkpoint['best_val_loss']
        
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"✅ 체크포인트 로드 완료: Epoch {start_epoch-1}")
        
        return start_epoch
    
    def train(self, num_epochs: int, resume_from: str = None, 
              save_every_n_steps: int = 500, log_every_n_steps: int = 100):
        """전체 학습 프로세스 (종료 신호 지원)"""
        start_epoch = 0
        
        # 체크포인트에서 재개
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        logger.info(f"🚀 학습 시작: Epoch {start_epoch} → {num_epochs}")
        logger.info("=" * 60)
        
        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                
                # 종료 요청 확인
                if self.shutdown_requested:
                    logger.info("⚠️ 종료 요청으로 인한 학습 중단")
                    break
                
                epoch_start_time = time.time()
                
                # 훈련
                train_loss, train_metrics = self.train_epoch(
                    epoch, save_every_n_steps, log_every_n_steps
                )
                
                # 조기 종료 확인
                if self.shutdown_requested:
                    logger.info("⚠️ 에포크 중 종료 요청 - 현재 상태 저장 후 종료")
                    self.save_checkpoint(epoch, is_best=False, suffix="emergency_save")
                    break
                
                # 검증
                val_loss, val_metrics = self.validate_epoch(epoch)
                
                # 스케줄러 업데이트
                self.scheduler.step(val_loss)
                
                # 로깅
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch:3d}/{num_epochs-1} ({epoch_time:.1f}s)")
                logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                logger.info(f"  Train Acc: {train_metrics.get('word_accuracy', 0):.3f} | "
                           f"Val Acc: {val_metrics.get('word_accuracy', 0):.3f}")
                
                # 최고 성능 모델 체크
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    logger.info(f"🎉 새로운 최고 성능! Val Loss: {val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                
                # 체크포인트 저장 (매 에포크마다 저장)
                self.save_checkpoint(epoch, is_best)
                
                # 얼리 스토핑
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"⏹️ 얼리 스토핑: {self.early_stopping_patience} 에포크 개선 없음")
                    break
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("-" * 60)
                
        except Exception as e:
            logger.error(f"❌ 학습 중 오류 발생: {e}")
            # 응급 체크포인트 저장
            self.save_checkpoint(self.current_epoch, is_best=False, suffix="error_checkpoint")
            raise
        
        finally:
            # 리소스 정리
            if hasattr(self, 'writer'):
                self.writer.close()
                logger.info("📝 TensorBoard writer 종료")
            
            if self.shutdown_requested:
                logger.info("✅ 안전한 종료 완료")
            else:
                logger.info("🎉 정상 학습 완료")
        
        # 학습 완료
        logger.info("🎉 학습 완료!")
        self.writer.close()
        
        return self.best_val_loss

def main():
    """메인 학습 함수"""
    print("🚀 수화 인식 모델 학습 시작")
    print("=" * 50)
    
    # 설정
    config = {
        'batch_size': 8,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'annotation_path': "./data/sign_language_dataset_only_sen_lzf.h5",
        'pose_data_dir': "./data",
        'sequence_length': 200,
        'min_segment_length': 20,
        'max_segment_length': 300,
        'train_split': 0.8,
        'resume_from': None  # 체크포인트 경로 (필요시)
    }
    
    print(f"⚙️ 설정:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # 데이터 로더 생성
        print(f"\n📊 데이터 로더 생성 중...")
        full_dataloader, dataset = create_dataloader(
            annotation_path=config['annotation_path'],
            pose_data_dir=config['pose_data_dir'],
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            min_segment_length=config['min_segment_length'],
            max_segment_length=config['max_segment_length'],
            shuffle=True
        )
        
        # 훈련/검증 분할
        dataset_size = len(dataset)
        train_size = int(config['train_split'] * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=full_dataloader.collate_fn,
            pin_memory=True
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            collate_fn=full_dataloader.collate_fn,
            pin_memory=True
        )
        
        print(f"✅ 데이터 로더 생성 완료:")
        print(f"   총 데이터: {dataset_size}")
        print(f"   훈련 데이터: {train_size}")
        print(f"   검증 데이터: {val_size}")
        print(f"   Vocabulary 크기: {dataset.vocab_size}")
        
        # 모델 생성
        print(f"\n🤖 모델 생성 중...")
        model = SequenceToSequenceSignModel(
            vocab_size=dataset.vocab_size,
            embed_dim=256,
            num_encoder_layers=6,
            num_decoder_layers=4,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        
        # 트레이너 생성
        trainer = SignLanguageTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=config['device'],
            vocab_words=dataset.words
        )
        
        # 학습 시작
        print(f"\n🏋️ 학습 시작...")
        best_loss = trainer.train(
            num_epochs=config['num_epochs'],
            resume_from=config['resume_from']
        )
        
        print(f"\n🎉 학습 완료!")
        print(f"   최고 성능: {best_loss:.4f}")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
