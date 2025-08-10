#!/usr/bin/env python3
"""
빠른 다단계 학습 테스트 (각 스테이지 1 에포크)
"""

import logging
from advanced_config import MultiStageTrainingConfig, TrainingStageConfig
from simple_advanced_trainer import SimpleAdvancedTrainer
from unified_pose_dataloader import UnifiedSignLanguageDataset, create_dataloader
from sign_language_model import SequenceToSequenceSignModel
from advanced_data_utils import create_dataloaders
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """빠른 다단계 학습 테스트"""
    
    logger.info("🚀 빠른 다단계 학습 테스트 시작!")
    
    # 설정
    config = MultiStageTrainingConfig()
    config.seed.fix_all_seeds()
    
    # 매우 짧은 스테이지들로 재정의 (각 1 에포크)
    config.stages = [
        TrainingStageConfig(
            name="baseline",
            description="기본 학습 (1 에포크)",
            epochs=1,
            batch_size=16,  # 배치 크기 증가로 속도 향상
            learning_rate=1e-4,
            dropout=0.1,
            enable_augmentation=False
        ),
        TrainingStageConfig(
            name="augmentation",
            description="데이터 증강 (1 에포크)",
            epochs=1,
            batch_size=16,
            learning_rate=5e-5,
            dropout=0.15,
            enable_augmentation=True,
            augmentation_strength=0.3
        )
    ]
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"💻 {device.type.upper()} 디바이스 사용")
    
    # 기본 데이터셋 로드
    logger.info("📊 기본 데이터셋 로드 중...")
    base_dataset = UnifiedSignLanguageDataset(
        annotation_file='./data/sign_language_dataset_only_sen_lzf.h5',
        pose_dir='./data',
        enable_augmentation=False
    )
    logger.info(f"✅ 기본 데이터셋 로드 완료: {len(base_dataset)}개 세그먼트")
    
    # 데이터 분할 및 로더 생성
    logger.info("📊 데이터 분할 및 로더 생성 중...")
    dataloaders = create_dataloaders(
        base_dataset,
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.05,
        batch_size=16,
        num_workers=2
    )
    logger.info(f"✅ 데이터 로더 생성 완료")
    
    # 모델 생성
    vocab_size = len(base_dataset.words)
    model = SequenceToSequenceSignModel(
        vocab_size=vocab_size,
        input_dim=133 * 3,  # MediaPipe keypoints
        hidden_dim=512,
        num_layers=4
    )
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 모델 생성: {total_params:,} 파라미터")
    
    # 트레이너 생성
    trainer = SimpleAdvancedTrainer(
        config=config,
        device=device
    )
    
    # 다단계 학습 실행
    logger.info("=" * 60)
    logger.info("🚀 빠른 다단계 학습 시작 (2단계, 각 1에포크)")
    logger.info("=" * 60)
    
    try:
        best_model = trainer.train_multi_stage(model, dataloaders)
        logger.info("✅ 다단계 학습 완료!")
        logger.info(f"📊 최종 모델 파라미터: {sum(p.numel() for p in best_model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"❌ 학습 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
