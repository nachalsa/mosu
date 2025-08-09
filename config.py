"""
수화 인식 시스템 통합 설정 파일
"""

import torch
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """모델 설정"""
    vocab_size: int = 442
    embed_dim: int = 384  # 256 -> 384 (통일)
    num_encoder_layers: int = 6  # 4 -> 6 (통일)
    num_decoder_layers: int = 4  # 3 -> 4 (통일)
    num_heads: int = 12  # 8 -> 12 (통일)
    dim_feedforward: int = 1536  # 1024 -> 1536 (통일)
    max_seq_len: int = 200
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """학습 설정"""
    batch_size: int = 32  # 4 -> 32 (8배)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 50
    warmup_steps: int = 500
    gradient_clip_val: float = 1.0
    
    # 손실 함수 가중치
    word_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.3
    
    # 스케줄러 설정
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6


@dataclass
class DataConfig:
    """데이터 설정"""
    annotation_path: str = "./data/sign_language_dataset_only_sen_lzf.h5"
    pose_data_dir: str = "./data"
    sequence_length: int = 150  # 통일
    min_segment_length: int = 10
    max_segment_length: int = 300
    
    # 데이터 분할
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # 필터링 설정
    valid_data_types: tuple = (1,)  # SEN 타입만
    valid_views: tuple = (0,)       # 정면(F)만
    valid_real_ids: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16)
    
    # 데이터 증강 설정
    enable_augmentation: bool = False
    augmentation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """기본 증강 설정 초기화"""
        if self.augmentation_config is None:
            self.augmentation_config = {
                'enable_horizontal_flip': True,
                'enable_rotation': True,
                'enable_scaling': True,
                'enable_noise': True,
                'horizontal_flip_prob': 0.5,
                'rotation_range': 10.0,        # 보수적 ±10도
                'scaling_range': (0.95, 1.05), # 보수적 ±5%
                'noise_std': 0.005              # 작은 노이즈
            }


@dataclass
class SystemConfig:
    """시스템 설정"""
    device: str = "auto"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every_n_steps: int = 999999  # 에포크 기반 저장을 위해 비활성화 (큰 값으로 설정)
    log_every_n_steps: int = 100
    num_workers: int = 4
    pin_memory: bool = True
    
    # 시드 설정
    random_seed: int = 42
    
    # 안전 종료 설정
    enable_signal_handler: bool = True
    graceful_shutdown_timeout: int = 30


class DeviceSpecificConfig:
    """디바이스별 설정"""
    
    @staticmethod
    def get_device_config(device_type: str = "auto") -> Dict[str, Any]:
        """디바이스 타입에 따른 최적화된 설정 반환"""
        
        # 자동 디바이스 감지
        if device_type == "auto":
            device_type = DeviceSpecificConfig.detect_device()
        
        configs = {
            "cuda": {
                "model": ModelConfig(
                    embed_dim=384,  # 512 -> 384 (통일)
                    num_encoder_layers=6,  # 8 -> 6 (통일)
                    num_decoder_layers=4,  # 6 -> 4 (통일)
                    num_heads=12,  # 16 -> 12 (통일)
                    dim_feedforward=1536,  # 2048 -> 1536 (통일)
                    max_seq_len=200
                ),
                "training": TrainingConfig(
                    batch_size=128,  # 16 -> 128 (8배)
                    learning_rate=2e-4,
                    num_epochs=100,
                    warmup_steps=1000
                ),
                "data": DataConfig(
                    sequence_length=150  # 200 -> 150 (통일)
                ),
                "system": SystemConfig(
                    device="cuda",
                    checkpoint_dir="./checkpoints_cuda",
                    log_dir="./logs_cuda",
                    num_workers=8
                )
            },
            "xpu": {
                "model": ModelConfig(
                    embed_dim=384,
                    num_encoder_layers=6,
                    num_decoder_layers=4,
                    num_heads=12,
                    dim_feedforward=1536,
                    max_seq_len=200
                ),
                "training": TrainingConfig(
                    batch_size=48,  # 6 -> 48 (8배)
                    learning_rate=2e-4,
                    num_epochs=75
                ),
                "data": DataConfig(
                    sequence_length=200  # 통일
                ),
                "system": SystemConfig(
                    device="xpu",
                    checkpoint_dir="./checkpoints_xpu",
                    log_dir="./logs_xpu",
                    num_workers=4
                )
            },
            "cpu": {
                "model": ModelConfig(
                    embed_dim=384,  # 256 -> 384 (통일)
                    num_encoder_layers=6,  # 4 -> 6 (통일)
                    num_decoder_layers=4,  # 3 -> 4 (통일)
                    num_heads=12,  # 8 -> 12 (통일)
                    dim_feedforward=1536,  # 1024 -> 1536 (통일)
                    max_seq_len=200  # 150 -> 200 (통일)
                ),
                "training": TrainingConfig(
                    batch_size=16,  # 2 -> 16 (8배)
                    learning_rate=1e-4,
                    num_epochs=30
                ),
                "data": DataConfig(
                    sequence_length=200  # 통일
                ),
                "system": SystemConfig(
                    device="cpu",
                    checkpoint_dir="./checkpoints_cpu",
                    log_dir="./logs_cpu",
                    num_workers=2,
                    pin_memory=False
                )
            }
        }
        
        return configs.get(device_type, configs["cpu"])
    
    @staticmethod
    def detect_device() -> str:
        """자동으로 최적 디바이스 감지"""
        # CUDA 확인
        if torch.cuda.is_available():
            return "cuda"
        
        # XPU 확인 (안전한 감지 로직)
        if torch.xpu.is_available():
            return "xpu"
        
        # CPU로 폴백
        return "cpu"
    
    @staticmethod
    def optimize_for_device(config: Dict[str, Any], device_type: str) -> Dict[str, Any]:
        """디바이스별 추가 최적화"""
        if device_type == "xpu":
            # XPU 특화 최적화
            config["training"].gradient_clip_val = 0.5
            config["model"].dropout = 0.15
            
        elif device_type == "cuda":
            # CUDA 특화 최적화
            config["system"].pin_memory = True
            config["training"].gradient_clip_val = 1.0
            
        elif device_type == "cpu":
            # CPU 특화 최적화
            config["system"].pin_memory = False
            config["system"].num_workers = min(2, os.cpu_count() or 1)
            config["model"].dropout = 0.05
        
        return config


def create_config(device_type: str = "auto", custom_overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """통합 설정 생성"""
    # 디바이스별 기본 설정 가져오기
    config = DeviceSpecificConfig.get_device_config(device_type)
    
    # 디바이스별 최적화 적용
    actual_device = config["system"].device
    config = DeviceSpecificConfig.optimize_for_device(config, actual_device)
    
    # 커스텀 오버라이드 적용
    if custom_overrides:
        for section, overrides in custom_overrides.items():
            if section in config:
                for key, value in overrides.items():
                    if hasattr(config[section], key):
                        setattr(config[section], key, value)
    
    return config


def print_config(config: Dict[str, Any]):
    """설정 정보 출력"""
    print("=" * 60)
    print("🔧 수화 인식 시스템 설정")
    print("=" * 60)
    
    device_type = config["system"].device
    print(f"🎯 디바이스: {device_type.upper()}")
    print(f"📊 배치 크기: {config['training'].batch_size}")
    print(f"🧠 모델 임베딩 차원: {config['model'].embed_dim}")
    print(f"🔢 인코더 레이어: {config['model'].num_encoder_layers}")
    print(f"📝 시퀀스 길이: {config['data'].sequence_length}")
    print(f"💾 체크포인트: {config['system'].checkpoint_dir}")
    print("=" * 60)


# 환경별 사전 정의된 설정들
CONFIGS = {
    "development": {
        "training": {"num_epochs": 10, "save_every_n_steps": 100},
        "model": {"embed_dim": 256}
    },
    "production": {
        "training": {"num_epochs": 100, "save_every_n_steps": 1000},
        "system": {"log_every_n_steps": 200}
    },
    "debug": {
        "training": {"batch_size": 1, "num_epochs": 2},
        "data": {"sequence_length": 50},
        "system": {"log_every_n_steps": 1}
    }
}


if __name__ == "__main__":
    # 설정 테스트
    print("🧪 설정 시스템 테스트")
    
    # 자동 감지
    auto_config = create_config("auto")
    print_config(auto_config)
    
    # 개발 환경 설정 테스트
    dev_config = create_config("auto", CONFIGS["development"])
    print(f"\n🔧 개발 환경 - 에폭 수: {dev_config['training'].num_epochs}")
