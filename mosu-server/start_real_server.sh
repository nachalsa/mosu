#!/bin/bash
echo "🚀 MOSU 실제 모델 서버 시작 스크립트"

# 가상환경 활성화
cd /home/lts/gitwork/mosu
source .venv_xpu/bin/activate

echo "✅ 가상환경 활성화됨"
echo "Python 경로: $(which python)"

# 필요한 패키지 확인
echo ""
echo "📦 패키지 확인 중..."
python -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} (디바이스: {\"CUDA\" if torch.cuda.is_available() else \"XPU\" if hasattr(torch, \"xpu\") and torch.xpu.is_available() else \"CPU\"})')
    
    import fastapi, uvicorn, numpy
    print('✅ FastAPI, Uvicorn, NumPy 사용 가능')
    
    from sign_language_model import SequenceToSequenceSignModel
    print('✅ 수화 인식 모델 모듈 로딩 가능')
    
except ImportError as e:
    print(f'❌ 패키지 오류: {e}')
    exit(1)
"

# 모델 파일 확인
if [ -f "mosumodel/best_model_stage_1.pt" ]; then
    echo "✅ 수화 모델 파일 존재 ($(du -h mosumodel/best_model_stage_1.pt | cut -f1))"
else
    echo "❌ 수화 모델 파일 없음: mosumodel/best_model_stage_1.pt"
    echo "더미 모델로 폴백됩니다."
fi

echo ""
echo "🚀 실제 MOSU 서버 실행 중..."
echo "   - 포트: 8002"
echo "   - 주소: http://192.168.100.26:8002"
echo "   - 네트워크: 192.168.100.26"
echo "   - 실제 Transformer 모델 사용"
echo "   - Ctrl+C로 종료"
echo ""

# 서버 실행
cd mosu-server
python real_server.py --model ../mosumodel/best_model_stage_1.pt --device auto --host 192.168.100.26 --port 8002
