"""
디바이스 자동 감지 및 설정 유틸리티
"""
import torch
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """통합 디바이스 관리자"""
    
    @staticmethod
    def detect_best_device(preferred_device: str = "auto") -> torch.device:
        """최적의 디바이스 자동 감지"""
        
        if preferred_device != "auto":
            # 사용자 지정 디바이스 검증
            return DeviceManager._validate_device(preferred_device)
        
        # 자동 감지 순서: XPU > CUDA > CPU
        logger.info("🔍 사용 가능한 디바이스 검색 중...")
        
        # 1. XPU 확인 (Intel GPU)
        xpu_device = DeviceManager._check_xpu()
        if xpu_device:
            return xpu_device
        
        # 2. CUDA 확인 (NVIDIA GPU)
        cuda_device = DeviceManager._check_cuda()
        if cuda_device:
            return cuda_device
        
        # 3. CPU 폴백
        logger.info("💻 CPU 디바이스 사용")
        return torch.device("cpu")
    
    @staticmethod
    def _check_xpu() -> torch.device:
        """XPU 사용 가능성 확인"""
        try:
            # XPU 모듈 존재 확인
            if not hasattr(torch, 'xpu'):
                logger.debug("XPU 모듈 미지원")
                return None
            
            # device_count() 방식으로 확인
            try:
                device_count = torch.xpu.device_count()
                if device_count > 0:
                    logger.info(f"⚡ XPU 디바이스 감지: {device_count}개 디바이스")
                    # 추가 검증: 실제 텐서 생성 테스트
                    test_tensor = torch.tensor([1.0], device='xpu:0')
                    logger.info(f"✅ XPU 디바이스 사용: {test_tensor.device}")
                    return torch.device("xpu:0")
            except Exception as e:
                logger.debug(f"XPU device_count() 실패: {e}")
            
            # is_available() 방식으로 확인 (있는 경우)
            try:
                if hasattr(torch.xpu, 'is_available') and torch.xpu.is_available():
                    test_tensor = torch.tensor([1.0], device='xpu')
                    logger.info(f"⚡ XPU 디바이스 사용: {test_tensor.device}")
                    return torch.device("xpu")
            except Exception as e:
                logger.debug(f"XPU is_available() 실패: {e}")
            
            # 직접 테스트 방식
            try:
                test_tensor = torch.tensor([1.0], device='xpu')
                logger.info(f"⚡ XPU 디바이스 사용: {test_tensor.device}")
                return torch.device("xpu")
            except Exception as e:
                logger.debug(f"XPU 직접 테스트 실패: {e}")
                
        except Exception as e:
            logger.debug(f"XPU 전체 확인 실패: {e}")
        
        return None
    
    @staticmethod
    def _check_cuda() -> torch.device:
        """CUDA 사용 가능성 확인"""
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                logger.info(f"🔥 CUDA 디바이스 사용: {device_name} ({device_count}개 디바이스)")
                return torch.device(f"cuda:{current_device}")
        except Exception as e:
            logger.debug(f"CUDA 확인 실패: {e}")
        
        return None
    
    @staticmethod
    def _validate_device(device_str: str) -> torch.device:
        """사용자 지정 디바이스 검증"""
        try:
            device = torch.device(device_str)
            
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("⚠️ CUDA 요청되었으나 사용 불가 - CPU로 폴백")
                    return torch.device("cpu")
                logger.info(f"🔥 지정된 CUDA 디바이스 사용: {device}")
                return device
            
            elif device.type == "xpu":
                xpu_device = DeviceManager._check_xpu()
                if xpu_device is None:
                    logger.warning("⚠️ XPU 요청되었으나 사용 불가 - CPU로 폴백")
                    return torch.device("cpu")
                logger.info(f"⚡ 지정된 XPU 디바이스 사용: {device}")
                return device
            
            elif device.type == "cpu":
                logger.info("💻 지정된 CPU 디바이스 사용")
                return device
            
            else:
                logger.warning(f"⚠️ 알 수 없는 디바이스 타입: {device} - CPU로 폴백")
                return torch.device("cpu")
                
        except Exception as e:
            logger.error(f"❌ 디바이스 검증 실패: {e} - CPU로 폴백")
            return torch.device("cpu")
    
    @staticmethod
    def get_device_info(device: torch.device) -> dict:
        """디바이스 정보 반환"""
        info = {
            'type': device.type,
            'index': device.index,
            'name': str(device)
        }
        
        try:
            if device.type == "cuda":
                info.update({
                    'name': torch.cuda.get_device_name(device),
                    'memory_total': torch.cuda.get_device_properties(device).total_memory,
                    'memory_cached': torch.cuda.memory_cached(device),
                    'memory_allocated': torch.cuda.memory_allocated(device)
                })
            elif device.type == "xpu":
                # XPU 정보 (가능한 경우)
                try:
                    if hasattr(torch.xpu, 'get_device_name'):
                        info['name'] = torch.xpu.get_device_name(device.index)
                except:
                    pass
            elif device.type == "cpu":
                import os
                info.update({
                    'cores': os.cpu_count(),
                    'threads': torch.get_num_threads()
                })
        except Exception as e:
            logger.debug(f"디바이스 정보 수집 실패: {e}")
        
        return info
    
    @staticmethod
    def optimize_for_device(device: torch.device):
        """디바이스별 최적화 설정"""
        if device.type == "cuda":
            # CUDA 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("🔥 CUDA 최적화 설정 완료")
            
        elif device.type == "xpu":
            # XPU 최적화 (가능한 경우)
            try:
                # XPU 관련 최적화 설정
                logger.info("⚡ XPU 최적화 설정 완료")
            except:
                pass
                
        elif device.type == "cpu":
            # CPU 최적화
            torch.set_num_threads(min(8, torch.get_num_threads()))
            logger.info("💻 CPU 최적화 설정 완료")

def get_device_string(device: torch.device) -> str:
    """디바이스를 문자열로 변환"""
    if device.index is not None:
        return f"{device.type}:{device.index}"
    return device.type

# 편의 함수들
def auto_device(preferred: str = "auto") -> torch.device:
    """자동 디바이스 선택"""
    return DeviceManager.detect_best_device(preferred)

def device_info(device: torch.device = None) -> dict:
    """디바이스 정보"""
    if device is None:
        device = auto_device()
    return DeviceManager.get_device_info(device)

def optimize_device(device: torch.device = None):
    """디바이스 최적화"""
    if device is None:
        device = auto_device()
    DeviceManager.optimize_for_device(device)

if __name__ == "__main__":
    # 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("=== 디바이스 관리자 테스트 ===")
    
    # 자동 감지
    device = auto_device()
    print(f"감지된 디바이스: {device}")
    
    # 정보 출력
    info = device_info(device)
    print(f"디바이스 정보: {info}")
    
    # 최적화 적용
    optimize_device(device)
    
    print("테스트 완료!")
