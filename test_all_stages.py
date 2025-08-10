#!/usr/bin/env python3
"""
전체 5단계 시스템 테스트 스크립트
"""
import sys
import logging
from advanced_config import AdvancedTrainingConfig, TrainingStageConfig
from advanced_trainer import AdvancedSignLanguageTrainer

def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    print("🧪 전체 5단계 시스템 테스트 시작")
    
    try:
        # 설정 생성
        config = AdvancedTrainingConfig()
        config.experiment_name = 'all_stages_test'
        
        # 각 단계를 매우 짧게 설정 (빠른 검증용)
        print("⚙️ 단계 설정 축소 중...")
        for i, stage in enumerate(config.multi_stage.stages):
            stage.num_epochs = 1  # 모든 단계를 1에포크로
            stage.batch_size = 16  # 배치 크기 축소
            print(f"   Stage {i+1} ({stage.name}): 1 epoch, batch_size=16")
        
        print(f"📋 총 {len(config.multi_stage.stages)}단계 설정 완료")
        
        # 트레이너 생성
        print("🚀 트레이너 초기화 중...")
        trainer = AdvancedSignLanguageTrainer(config)
        
        # 전체 단계 실행
        print("📊 전체 단계 실행 시작...")
        results = trainer.train_multi_stage()
        
        print("🎉 전체 5단계 테스트 성공!")
        print(f"✅ 완료된 단계 수: {len(results.get('stages', []))}")
        
        # 각 단계별 결과 간략히 출력
        for i, stage_result in enumerate(results.get('stages', [])):
            print(f"   Stage {i+1} ({stage_result['stage_name']}): "
                  f"Val Acc {stage_result['best_val_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
