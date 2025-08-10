#!/usr/bin/env python3
"""
고급 학습 시스템 간단 테스트
"""
import os
import logging
import sys
from pathlib import Path

# 현재 경로 추가
sys.path.append(str(Path(__file__).parent))

from advanced_config import AdvancedTrainingConfig, TrainingStageConfig
from advanced_data_utils import StratifiedDataSplitter
from unified_pose_dataloader import UnifiedSignLanguageDataset

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def test_config():
    """설정 테스트"""
    print("🧪 1. 설정 테스트")
    config = AdvancedTrainingConfig()
    config.random_seed.fix_all_seeds()
    
    print(f"✅ 랜덤시드: {config.random_seed.seed}")
    print(f"✅ 학습 단계 수: {len(config.multi_stage.stages)}")
    
    for i, stage in enumerate(config.multi_stage.stages):
        print(f"  Stage {i+1}: {stage.name} ({stage.num_epochs} epochs)")
    
    return config

def test_dataset():
    """데이터셋 테스트"""
    print("\n🧪 2. 데이터셋 테스트")
    
    try:
        dataset = UnifiedSignLanguageDataset(
            annotation_path="./data/sign_language_dataset_only_sen_lzf.h5",
            pose_data_dir="./data",
            sequence_length=200,
            min_segment_length=20,
            max_segment_length=300,
            enable_augmentation=False
        )
        
        print(f"✅ 데이터셋 로드 성공: {len(dataset)}개 세그먼트")
        print(f"✅ Vocabulary 크기: {dataset.vocab_size}")
        print(f"✅ vocab 속성 확인: {hasattr(dataset, 'vocab')}")
        
        if hasattr(dataset, 'vocab') and len(dataset.vocab) > 0:
            first_words = list(dataset.vocab.values())[:5]
            print(f"✅ 첫 5개 단어: {first_words}")
        
        # 샘플 데이터 테스트
        sample = dataset[0]
        print(f"✅ 샘플 데이터 형태:")
        print(f"  pose_features: {sample['pose_features'].shape}")
        print(f"  vocab_ids: {sample['vocab_ids'].shape}")
        print(f"  duration: {sample['duration']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return None

def test_data_splitter(dataset):
    """데이터 분할 테스트"""
    print("\n🧪 3. 데이터 분할 테스트")
    
    try:
        config = AdvancedTrainingConfig()
        splitter = StratifiedDataSplitter(
            config=config.data_split,
            random_seed=config.random_seed.seed
        )
        
        train_indices, val_indices, test_indices = splitter.split_dataset_stratified(dataset)
        
        print(f"✅ 데이터 분할 성공:")
        print(f"  훈련: {len(train_indices)}개")
        print(f"  검증: {len(val_indices)}개") 
        print(f"  테스트: {len(test_indices)}개")
        print(f"  총합: {len(train_indices) + len(val_indices) + len(test_indices)}개")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 분할 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloaders(dataset):
    """데이터로더 테스트"""
    print("\n🧪 4. 데이터로더 테스트")
    
    try:
        config = AdvancedTrainingConfig()
        splitter = StratifiedDataSplitter(
            config=config.data_split,
            random_seed=config.random_seed.seed
        )
        
        train_loader, val_loader, test_loader = splitter.create_dataloaders(
            dataset=dataset,
            batch_size=4,
            enable_train_augmentation=False,
            augmentation_config=None
        )
        
        print(f"✅ 데이터로더 생성 성공")
        
        # 배치 테스트
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"✅ 배치 테스트:")
        print(f"  훈련 배치: {train_batch['pose_features'].shape}")
        print(f"  검증 배치: {val_batch['pose_features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터로더 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트"""
    print("🚀 고급 학습 시스템 간단 테스트 시작")
    print("=" * 60)
    
    setup_logging()
    
    # 1. 설정 테스트
    config = test_config()
    
    # 2. 데이터셋 테스트
    dataset = test_dataset()
    if dataset is None:
        print("❌ 데이터셋 테스트 실패로 종료")
        return False
    
    # 3. 데이터 분할 테스트
    split_success = test_data_splitter(dataset)
    if not split_success:
        print("❌ 데이터 분할 테스트 실패로 종료")
        return False
    
    # 4. 데이터로더 테스트
    loader_success = test_dataloaders(dataset)
    if not loader_success:
        print("❌ 데이터로더 테스트 실패로 종료")
        return False
    
    print("\n🎉 모든 기본 테스트 통과!")
    print("고급 학습 시스템이 정상적으로 작동할 준비가 되었습니다.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
