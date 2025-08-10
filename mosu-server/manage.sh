#!/bin/bash

echo "🤲 MOSU 수화 인식 서버 관리 도구"
echo "=================================="
echo ""

print_menu() {
    echo "사용 가능한 서버:"
    echo "1. 간단한 테스트 서버 (포트: 8001) - 더미 모델, 빠른 테스트용"
    echo "2. 실제 모델 서버 (포트: 8002) - Transformer 모델, 실제 수화 인식"
    echo "3. 서버 상태 확인"
    echo "4. 모든 서버 종료"
    echo "5. 종료"
    echo ""
}

check_server_status() {
    echo "🔍 서버 상태 확인 중..."
    echo ""
    
    # 포트 8001 확인
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then
        echo "✅ 간단한 테스트 서버 (8001): 실행 중"
        echo "   📊 상태: $(curl -s http://localhost:8001/health | jq -r '.status // "unknown"')"
    else
        echo "❌ 간단한 테스트 서버 (8001): 중지됨"
    fi
    
    # 포트 8002 확인
    if curl -s http://localhost:8002/health >/dev/null 2>&1; then
        echo "✅ 실제 모델 서버 (8002): 실행 중"
        echo "   📊 상태: $(curl -s http://localhost:8002/health | jq -r '.status // "unknown"')"
        echo "   🧠 모델: $(curl -s http://localhost:8002/health | jq -r '.model_type // "unknown"')"
    else
        echo "❌ 실제 모델 서버 (8002): 중지됨"
    fi
    
    echo ""
}

kill_servers() {
    echo "🔄 모든 MOSU 서버 종료 중..."
    
    # Python 프로세스 중 MOSU 관련 서버 종료
    pkill -f "simple_server.py" 2>/dev/null && echo "✅ 간단한 서버 종료됨"
    pkill -f "real_server.py" 2>/dev/null && echo "✅ 실제 서버 종료됨"
    pkill -f "test_server.py" 2>/dev/null && echo "✅ 테스트 서버 종료됨"
    
    sleep 2
    echo "🏁 정리 완료"
}

start_simple_server() {
    echo "🚀 간단한 테스트 서버 시작 중..."
    echo "   - 포트: 8001"
    echo "   - 더미 모델 사용"
    echo "   - 주소: http://localhost:8001"
    echo ""
    
    cd /home/lts/gitwork/mosu/mosu-server
    ./start_server.sh
}

start_real_server() {
    echo "🚀 실제 모델 서버 시작 중..."
    echo "   - 포트: 8002" 
    echo "   - Transformer 모델 사용"
    echo "   - 주소: http://localhost:8002"
    echo ""
    
    cd /home/lts/gitwork/mosu/mosu-server
    ./start_real_server.sh
}

# 메인 루프
while true; do
    print_menu
    read -p "선택하세요 (1-5): " choice
    echo ""
    
    case $choice in
        1)
            start_simple_server
            break
            ;;
        2)
            start_real_server
            break
            ;;
        3)
            check_server_status
            read -p "계속하려면 Enter를 누르세요..."
            clear
            ;;
        4)
            kill_servers
            read -p "계속하려면 Enter를 누르세요..."
            clear
            ;;
        5)
            echo "👋 MOSU 관리 도구를 종료합니다."
            exit 0
            ;;
        *)
            echo "❌ 잘못된 선택입니다. 1-5 사이의 숫자를 입력하세요."
            echo ""
            ;;
    esac
done
