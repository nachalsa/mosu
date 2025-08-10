#!/bin/bash

# MOSU 시스템 통합 시작 스크립트
# 분산 배포를 위한 설정

echo "🚀 MOSU 시스템 시작"
echo "=================================="
echo "🌐 웹서버: 192.168.100.90:8000"
echo "🤖 통합백엔드: 192.168.100.26:8001"
echo "=================================="

# 현재 서버 확인
CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "📍 현재 서버 IP: $CURRENT_IP"

activate_venv() {
    if [ -f "/home/lts/gitwork/mosu/.venv_xpu/bin/activate" ]; then
        echo "🔧 가상환경 활성화 중..."
        source /home/lts/gitwork/mosu/.venv_xpu/bin/activate
        echo "✅ 가상환경 활성화 완료"
    else
        echo "⚠️  가상환경을 찾을 수 없습니다: /home/lts/gitwork/mosu/.venv_xpu/bin/activate"
    fi
}

activate_venv

# 서버별 실행
if [[ "$CURRENT_IP" == "192.168.100.90" ]]; then
    echo "🌐 웹서버 시작 중..."
    cd /home/lts/gitwork/mosu/web-main/app
    exec uvicorn main:app --host 0.0.0.0 --port 8000
    
elif [[ "$CURRENT_IP" == "192.168.100.26" ]]; then
    echo "🤖 통합 백엔드 서버 시작 중..."
    cd /home/lts/gitwork/mosu/full-server
    exec python3 full_server.py --host 0.0.0.0 --port 8000
    
else
    echo "⚠️ 알 수 없는 서버 IP: $CURRENT_IP"
    echo "사용 가능한 실행 옵션:"
    echo "  $0 web      - 웹서버 실행 (192.168.100.90)"
    echo "  $0 backend  - 통합백엔드 실행 (192.168.100.26)" 
    echo "  $0 local    - 로컬 테스트 (모든 서비스 localhost)"
    
    # 매뉴얼 실행 옵션
    case "$1" in
        web)
            echo "🌐 웹서버 매뉴얼 시작"
            cd /home/lts/gitwork/mosu/web-main/app
            exec uvicorn main:app --host 0.0.0.0 --port 8000
            ;;
        backend)
            echo "🤖 통합백엔드 매뉴얼 시작"
            cd /home/lts/gitwork/mosu/full-server
            exec python3 full_server.py --host 0.0.0.0 --port 8000
            ;;
        local)
            echo "🏠 로컬 테스트 모드"
            echo "   웹서버: http://localhost:8001"
            echo "   통합백엔드: http://localhost:8000"
            
            # 로컬에서 모든 서비스 시작
            cd /home/lts/gitwork/mosu
            
            # 백그라운드에서 통합백엔드 시작
            echo "🤖 통합백엔드 시작..."
            cd full-server && python3 full_server.py --host localhost --port 8000 &
            BACKEND_PID=$!
            
            # 웹서버 시작 (포그라운드)
            echo "🌐 웹서버 시작..."
            cd ../web-main/app
            uvicorn main:app --host localhost --port 8001
            
            # 종료 시 백그라운드 프로세스 정리
            trap "kill $BACKEND_PID 2>/dev/null" EXIT
            ;;
        *)
            echo "❌ 잘못된 옵션: $1"
            ;;
    esac
fi
