#!/bin/bash
echo "🚀 MOSU 서버 시작 스크립트"

# 가상환경 활성화
cd /home/lts/gitwork/mosu
source .venv_xpu/bin/activate

echo "✅ 가상환경 활성화됨"
echo "Python 경로: $(which python)"
echo "현재 디렉토리: $(pwd)"

# 필요한 패키지 확인
echo ""
echo "📦 패키지 확인 중..."
python -c "
try:
    import fastapi, uvicorn, numpy, torch
    print('✅ 모든 필수 패키지 사용 가능')
except ImportError as e:
    print(f'❌ 패키지 오류: {e}')
    exit(1)
"

# 모델 파일 확인
if [ -f "mosumodel/best_model_stage_1.pt" ]; then
    echo "✅ 수화 모델 파일 존재"
else
    echo "⚠️  수화 모델 파일 없음 (더미 모드로 실행)"
fi

echo ""
echo "🚀 MOSU 서버 실행 중..."
echo "   - 포트: 8001"
echo "   - 주소: http://192.168.100.26:8001"
echo "   - 네트워크: 192.168.100.26"
echo "   - Ctrl+C로 종료"
echo ""

# 서버 실행
cd mosu-server
python simple_server.py --host 192.168.100.26 --port 8001
