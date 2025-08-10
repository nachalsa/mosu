#!/usr/bin/env python3
"""
멀티 GPU 기능 테스트 스크립트
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from device_utils import DeviceManager
from advanced_config import AdvancedTrainingConfig
from sign_language_model import SequenceToSequenceSignModel

def test_device_detection():
    """디바이스 감지 테스트"""
    print("=" * 60)
    print("🔍 디바이스 감지 테스트")
    print("=" * 60)
    
    # 싱글 GPU 감지
    device_single = DeviceManager.detect_best_device("auto", multi_gpu=False)
    device_info_single = DeviceManager.get_device_info(device_single)
    print(f"싱글 GPU 모드: {device_info_single}")
    
    # 멀티 GPU 감지
    device_multi = DeviceManager.detect_best_device("auto", multi_gpu=True)
    device_info_multi = DeviceManager.get_device_info(device_multi)
    print(f"멀티 GPU 모드: {device_info_multi}")
    
    # 멀티 GPU 가용성 확인
    multi_gpu_available = DeviceManager.is_multi_gpu_available()
    print(f"멀티 GPU 사용 가능: {multi_gpu_available}")
    
    if multi_gpu_available:
        print(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

def test_multi_gpu_model():
    """멀티 GPU 모델 설정 테스트"""
    print("\n" + "=" * 60)
    print("🚀 멀티 GPU 모델 설정 테스트")
    print("=" * 60)
    
    # 간단한 모델 생성
    model = SequenceToSequenceSignModel(
        vocab_size=100,
        embed_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    device = DeviceManager.detect_best_device("auto", multi_gpu=True)
    print(f"기본 디바이스: {device}")
    
    model = model.to(device)
    print(f"모델이 {device}로 이동됨")
    
    # 멀티 GPU 설정 테스트
    if DeviceManager.is_multi_gpu_available() and torch.cuda.device_count() > 1:
        print("멀티 GPU 설정 시도...")
        try:
            multi_gpu_model = DeviceManager.setup_multi_gpu(model)
            print(f"✅ 멀티 GPU 설정 성공: {type(multi_gpu_model)}")
            print(f"DataParallel device_ids: {getattr(multi_gpu_model, 'device_ids', None)}")
        except Exception as e:
            print(f"❌ 멀티 GPU 설정 실패: {e}")
    else:
        print("멀티 GPU를 사용할 수 없거나 GPU가 1개뿐입니다.")

def test_batch_size_optimization():
    """배치 크기 최적화 테스트"""
    print("\n" + "=" * 60)
    print("📊 배치 크기 최적화 테스트")
    print("=" * 60)
    
    base_batch_size = 32
    device = DeviceManager.detect_best_device("auto", multi_gpu=True)
    
    effective_batch_size = DeviceManager.get_effective_batch_size(base_batch_size, device)
    print(f"기본 배치 크기: {base_batch_size}")
    print(f"최적화된 배치 크기: {effective_batch_size}")
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        expected_size = base_batch_size * torch.cuda.device_count()
        print(f"예상 배치 크기 (GPU 수 × 기본): {expected_size}")

def test_configuration():
    """멀티 GPU 설정 테스트"""
    print("\n" + "=" * 60)
    print("⚙️ 멀티 GPU 설정 테스트")
    print("=" * 60)
    
    # 멀티 GPU 설정으로 구성 생성
    config = AdvancedTrainingConfig(
        experiment_name="multi_gpu_test",
        multi_gpu=True,
        use_data_parallel=True,
        auto_adjust_batch_size=True
    )
    
    print(f"멀티 GPU 활성화: {config.multi_gpu}")
    print(f"DataParallel 사용: {config.use_data_parallel}")
    print(f"자동 배치 크기 조정: {config.auto_adjust_batch_size}")

if __name__ == "__main__":
    print("🧪 멀티 GPU 기능 테스트 시작")
    
    try:
        test_device_detection()
        test_multi_gpu_model()
        test_batch_size_optimization()
        test_configuration()
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
