#!/usr/bin/env python3
"""
멀티 GPU 학습 예제 스크립트
한국 수화 인식 모델을 멀티 GPU로 학습하는 방법을 보여줍니다.
"""

import torch
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_config import AdvancedTrainingConfig
from advanced_trainer import AdvancedSignLanguageTrainer

def create_multi_gpu_config():
    """멀티 GPU 학습을 위한 최적화된 설정 생성"""
    
    # GPU 수에 따른 배치 크기 조정
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    base_batch_size = 16  # 기본 배치 크기
    optimized_batch_size = base_batch_size * max(1, gpu_count)
    
    print(f"🔍 감지된 GPU 수: {gpu_count}")
    print(f"📊 최적화된 배치 크기: {optimized_batch_size}")
    
    config = AdvancedTrainingConfig(
        # 실험 설정
        experiment_name="multi_gpu_sign_language",
        
        # 멀티 GPU 설정
        multi_gpu=True,              # 멀티 GPU 모드 활성화
        use_data_parallel=True,      # DataParallel 사용
        auto_adjust_batch_size=True, # 자동 배치 크기 조정
        
        # 기본 학습 설정
        device="auto",               # 자동 디바이스 감지
        
        # 다단계 학습 설정
        multi_stage=AdvancedTrainingConfig.MultiStageConfig(
            stages=[
                # Stage 1: 기본 학습 (멀티 GPU 최적화)
                AdvancedTrainingConfig.TrainingStageConfig(
                    name="baseline_multi_gpu",
                    epochs=10,
                    batch_size=optimized_batch_size,
                    learning_rate=0.001,
                    enable_augmentation=False,
                    patience=5,
                    min_delta=0.001
                ),
                # Stage 2: 데이터 증강 학습
                AdvancedTrainingConfig.TrainingStageConfig(
                    name="augmentation_multi_gpu", 
                    epochs=15,
                    batch_size=optimized_batch_size,
                    learning_rate=0.0005,
                    enable_augmentation=True,
                    patience=7,
                    min_delta=0.0005
                ),
                # Stage 3: 세밀한 조정
                AdvancedTrainingConfig.TrainingStageConfig(
                    name="fine_tuning_multi_gpu",
                    epochs=20,
                    batch_size=optimized_batch_size // 2,  # 세밀한 조정을 위해 배치 크기 감소
                    learning_rate=0.0001,
                    enable_augmentation=True,
                    patience=10,
                    min_delta=0.0001,
                    weight_decay=0.01
                )
            ],
            max_stages=3,
            min_improvement_threshold=0.001,
            patience_between_stages=3
        ),
        
        # 모델 설정 (멀티 GPU에 최적화)
        model=AdvancedTrainingConfig.ModelConfig(
            hidden_dim=512,      # 멀티 GPU에서 더 큰 모델 사용 가능
            num_layers=8,        # 깊은 네트워크
            num_heads=8,
            dropout=0.1,
            max_seq_length=60
        ),
        
        # 데이터 설정
        data=AdvancedTrainingConfig.DataConfig(
            data_dir="data",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            min_sequence_length=5,
            max_sequence_length=60
        ),
        
        # 최적화 설정
        optimization=AdvancedTrainingConfig.OptimizationConfig(
            optimizer_type="adamw",
            weight_decay=0.001,
            gradient_clip_norm=1.0,
            warmup_steps=500,
            scheduler_type="cosine"
        ),
        
        # 체크포인트 및 로깅
        save_top_k=3,
        save_every_n_epochs=5,
        log_every_n_steps=50,
        
        # 재현성
        deterministic_training=True,
        random_seed=42
    )
    
    return config

def run_multi_gpu_training():
    """멀티 GPU 학습 실행"""
    
    print("🚀 멀티 GPU 한국 수화 인식 모델 학습 시작")
    print("=" * 80)
    
    # 멀티 GPU 사용 가능성 확인
    if not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    elif torch.cuda.device_count() == 1:
        print("ℹ️ GPU가 1개만 감지되었습니다. 싱글 GPU 모드로 실행됩니다.")
    else:
        print(f"✅ {torch.cuda.device_count()}개의 GPU가 감지되었습니다.")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 설정 생성
    config = create_multi_gpu_config()
    
    # 트레이너 초기화
    print("\n📋 트레이너 초기화 중...")
    trainer = AdvancedSignLanguageTrainer(config)
    
    # 학습 실행
    print("\n🎯 학습 시작...")
    try:
        results = trainer.train()
        
        print("\n✅ 학습 완료!")
        print("=" * 80)
        
        # 결과 요약
        if results:
            best_accuracy = max(stage['best_val_accuracy'] for stage in results)
            print(f"🏆 최고 검증 정확도: {best_accuracy:.4f}")
            
            total_epochs = sum(stage['epochs_completed'] for stage in results)
            print(f"📊 총 학습 에포크: {total_epochs}")
            
            print(f"📁 모델 저장 위치: {trainer.checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def show_multi_gpu_tips():
    """멀티 GPU 사용 팁"""
    print("\n" + "=" * 80)
    print("💡 멀티 GPU 학습 팁")
    print("=" * 80)
    
    tips = [
        "1. 배치 크기는 GPU 수에 비례해서 증가시키세요 (GPU 2개 = 배치 크기 2배)",
        "2. 학습률도 배치 크기에 맞춰 조정하는 것이 좋습니다",
        "3. 메모리 부족 시 --auto-adjust-batch-size 옵션을 사용하세요",
        "4. 모든 GPU가 동일한 메모리를 가지는지 확인하세요",
        "5. DataParallel 사용 시 첫 번째 GPU에 더 많은 메모리가 필요합니다",
        "6. 큰 모델일수록 멀티 GPU 효과가 더 큽니다"
    ]
    
    for tip in tips:
        print(f"  {tip}")

if __name__ == "__main__":
    print("🧪 멀티 GPU 학습 예제")
    
    # 커맨드라인 인자 처리
    import argparse
    parser = argparse.ArgumentParser(description="멀티 GPU 한국 수화 인식 모델 학습")
    parser.add_argument("--tips", action="store_true", help="멀티 GPU 사용 팁 표시")
    parser.add_argument("--test-only", action="store_true", help="테스트만 실행 (학습 안함)")
    
    args = parser.parse_args()
    
    if args.tips:
        show_multi_gpu_tips()
    
    if args.test_only:
        print("🧪 멀티 GPU 테스트 모드")
        # 여기서 test_multi_gpu.py의 기능을 호출할 수 있음
    else:
        run_multi_gpu_training()
        show_multi_gpu_tips()
