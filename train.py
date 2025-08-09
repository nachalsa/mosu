"""
통합 수화 인식 모델 학습 스크립트
XPU, CUDA, CPU 환경 자동 감지 및 최적화
"""

import os
import sys
import time
import signal
import traceback
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 모듈들
from config import create_config, print_config, CONFIGS
from unified_pose_dataloader import UnifiedSignLanguageDataset, collate_fn
from sign_language_model import SequenceToSequenceSignModel
from sign_language_trainer import SignLanguageTrainer


class GracefulShutdownHandler:
    """안전한 종료 신호 처리기"""
    
    def __init__(self):
        self.shutdown_requested = False
        self.trainer = None
        
    def register_trainer(self, trainer):
        """트레이너 등록"""
        self.trainer = trainer
        
    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        signal_names = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM"
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")
        
        print(f"\n⚠️ {signal_name} 수신됨. 안전하게 종료 중...")
        self.shutdown_requested = True
        
        if self.trainer:
            self.trainer.request_shutdown()
            
    def setup_signals(self):
        """시그널 핸들러 설정"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        print("✅ 종료 신호 핸들러 설정 완료")


def setup_logging(log_dir: str, device_type: str) -> logging.Logger:
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{device_type}_{timestamp}.log")
    
    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"로깅 시작: {log_file}")
    return logger


def load_data_with_retry(config: Dict[str, Any], logger: logging.Logger, max_retries: int = 3):
    """데이터 로딩 (재시도 로직 포함)"""
    data_config = config["data"]
    system_config = config["system"]
    
    for attempt in range(max_retries):
        try:
            logger.info(f"데이터셋 로딩 시도 {attempt + 1}/{max_retries}")
            
            # 기본 데이터셋 생성 (증강 비활성화)
            base_dataset = UnifiedSignLanguageDataset(
                annotation_path=data_config.annotation_path,
                pose_data_dir=data_config.pose_data_dir,
                sequence_length=data_config.sequence_length,
                min_segment_length=data_config.min_segment_length,
                max_segment_length=data_config.max_segment_length,
                enable_augmentation=False  # 기본 데이터셋 (증강 없음)
            )
            
            if len(base_dataset) == 0:
                raise ValueError("유효한 데이터가 없습니다.")
            
            logger.info(f"✅ 데이터셋 로딩 성공: {len(base_dataset)}개 세그먼트")
            
            # 훈련/검증 분할 (기본 데이터셋 기준)
            train_size = int(data_config.train_ratio * len(base_dataset))
            val_size = len(base_dataset) - train_size
            
            train_indices, val_indices = torch.utils.data.random_split(
                range(len(base_dataset)), [train_size, val_size],
                generator=torch.Generator().manual_seed(system_config.random_seed)
            )
            
            # 훈련용: 증강 활성화된 데이터셋에서 train_indices 사용
            if data_config.enable_augmentation:
                train_dataset_full = UnifiedSignLanguageDataset(
                    annotation_path=data_config.annotation_path,
                    pose_data_dir=data_config.pose_data_dir,
                    sequence_length=data_config.sequence_length,
                    min_segment_length=data_config.min_segment_length,
                    max_segment_length=data_config.max_segment_length,
                    enable_augmentation=True,
                    augmentation_config=data_config.augmentation_config
                )
                train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
            else:
                train_dataset = torch.utils.data.Subset(base_dataset, train_indices.indices)
            
            # 검증용: 항상 증강 없음
            val_dataset = torch.utils.data.Subset(base_dataset, val_indices.indices)
            
            # 데이터로더 생성
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["training"].batch_size,
                shuffle=True,
                num_workers=system_config.num_workers,
                pin_memory=system_config.pin_memory,
                collate_fn=collate_fn,
                persistent_workers=system_config.num_workers > 0
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config["training"].batch_size,
                shuffle=False,
                num_workers=system_config.num_workers,
                pin_memory=system_config.pin_memory,
                collate_fn=collate_fn,
                persistent_workers=system_config.num_workers > 0
            )
            
            logger.info(f"훈련 세트: {len(train_dataset)} ({'증강 활성화' if data_config.enable_augmentation else '증강 비활성화'}), 검증 세트: {len(val_dataset)} (증강 비활성화)")
            return train_dataloader, val_dataloader, base_dataset.vocab_size
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패 (시도 {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error("최대 재시도 횟수 초과. 데이터 로딩 실패")
                raise
            
            logger.info("잠시 후 재시도...")

            time.sleep(2)


def create_model(config: Dict[str, Any], vocab_size: int, logger: logging.Logger) -> nn.Module:
    """모델 생성"""
    model_config = config["model"]
    
    # vocab_size 업데이트
    model_config.vocab_size = vocab_size
    
    model = SequenceToSequenceSignModel(
        vocab_size=model_config.vocab_size,
        embed_dim=model_config.embed_dim,
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        num_heads=model_config.num_heads,
        dim_feedforward=model_config.dim_feedforward,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout
    )
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 모델 생성 완료")
    logger.info(f"   - 총 파라미터: {total_params:,}")
    logger.info(f"   - 학습 가능 파라미터: {trainable_params:,}")
    logger.info(f"   - 어휘 크기: {vocab_size}")
    
    return model


def setup_device_specific_optimizations(device_type: str, logger: logging.Logger):
    """디바이스별 최적화 설정"""
    if device_type == "xpu":
        if torch.xpu.is_available():
            logger.info(f"✅ XPU 디바이스: {torch.xpu.get_device_name()}")
            logger.info(f"   - 디바이스 수: {torch.xpu.device_count()}")
            if torch.xpu.device_count() > 0:
                props = torch.xpu.get_device_properties(0)
                logger.info(f"   - 메모리: {props.total_memory / 1e9:.1f}GB")
        else:
            logger.warning("⚠️ XPU 사용 불가, CPU로 폴백")

    elif device_type == "cuda":
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA 디바이스: {torch.cuda.get_device_name()}")
            logger.info(f"   - CUDA 버전: {torch.version.cuda}")
            logger.info(f"   - 디바이스 수: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                props = torch.cuda.get_device_properties(0)
                logger.info(f"   - 메모리: {props.total_memory / 1e9:.1f}GB")
        else:
            logger.warning("⚠️ CUDA 사용 불가, CPU로 폴백")
    
    elif device_type == "cpu":
        logger.info(f"✅ CPU 디바이스 사용")
        logger.info(f"   - CPU 스레드: {torch.get_num_threads()}")
        logger.info(f"   - CPU 코어: {os.cpu_count()}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="수화 인식 모델 학습")
    parser.add_argument("--device", choices=["auto", "cuda", "xpu", "cpu"], 
                       default="auto", help="사용할 디바이스")
    parser.add_argument("--config", choices=["development", "production", "debug"],
                       default=None, help="사전 정의된 설정")
    parser.add_argument("--resume", type=str, default=None,
                       help="재개할 체크포인트 경로")
    parser.add_argument("--epochs", type=int, default=None,
                       help="학습 에폭 수 (설정 오버라이드)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="배치 크기 (설정 오버라이드)")
    parser.add_argument("--no-signal-handler", action="store_true",
                       help="종료 신호 핸들러 비활성화")
    
    args = parser.parse_args()
    
    # 설정 생성
    custom_overrides = {}
    if args.config and args.config in CONFIGS:
        custom_overrides = CONFIGS[args.config]
    
    # 명령행 오버라이드
    if args.epochs:
        custom_overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        custom_overrides.setdefault("training", {})["batch_size"] = args.batch_size
    
    config = create_config(args.device, custom_overrides)
    
    # 시그널 핸들러 설정 오버라이드
    if args.no_signal_handler:
        config["system"].enable_signal_handler = False
    
    # 설정 출력
    print_config(config)
    
    # 시드 설정
    torch.manual_seed(config["system"].random_seed)
    
    # 로깅 설정
    logger = setup_logging(config["system"].log_dir, config["system"].device)
    
    # 종료 신호 핸들러 설정
    shutdown_handler = None
    if config["system"].enable_signal_handler:
        shutdown_handler = GracefulShutdownHandler()
        shutdown_handler.setup_signals()
    
    try:
        # 디바이스 최적화 설정
        setup_device_specific_optimizations(config["system"].device, logger)
        
        # 데이터 로딩
        logger.info("📊 데이터 로딩 시작...")
        train_dataloader, val_dataloader, vocab_size = load_data_with_retry(
            config, logger
        )
        
        # 모델 생성
        logger.info("🧠 모델 생성 시작...")
        model = create_model(config, vocab_size, logger)
        
        # 트레이너 설정
        logger.info("🎯 트레이너 설정 시작...")
        trainer = SignLanguageTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=config["system"].device,
            checkpoint_dir=config["system"].checkpoint_dir,
            log_dir=config["system"].log_dir
        )
        
        # 종료 핸들러에 트레이너 등록
        if shutdown_handler:
            shutdown_handler.register_trainer(trainer)
        
        # 학습 파라미터 설정
        trainer.setup_training(
            learning_rate=config["training"].learning_rate,
            weight_decay=config["training"].weight_decay,
            warmup_steps=config["training"].warmup_steps,
            gradient_clip_val=config["training"].gradient_clip_val,
            word_loss_weight=config["training"].word_loss_weight,
            boundary_loss_weight=config["training"].boundary_loss_weight,
            confidence_loss_weight=config["training"].confidence_loss_weight
        )
        
        # 학습 시작
        logger.info("🚀 학습 시작!")
        trainer.train(
            num_epochs=config["training"].num_epochs,
            resume_from=args.resume,
            save_every_n_steps=config["system"].save_every_n_steps,
            log_every_n_steps=config["system"].log_every_n_steps
        )
        
        logger.info("✅ 학습 완료!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 사용자에 의한 중단")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ 학습 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
        
    finally:
        logger.info("🔄 리소스 정리 중...")
        
        # 디바이스별 메모리 정리
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA 메모리 정리 완료")
        except:
            pass
        
        try:
            # XPU 메모리 정리 (가능한 경우)
            if hasattr(torch, 'xpu') and torch.xpu.device_count() > 0:
                torch.xpu.empty_cache()
                logger.info("XPU 메모리 정리 완료")
        except:
            pass


if __name__ == "__main__":
    main()
