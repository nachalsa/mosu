#!/bin/bash

echo "🌐 MOSU 시스템 네트워크 연결 상태 확인"
echo "========================================"

# 서버 정보
WEBSERVER="192.168.100.90:8000"
POSE_SERVER="192.168.100.135:5000"
MOSU_SERVER="192.168.100.26:8002"

echo -e "\n📋 서버 구성:"
echo "  🌐 웹 서버:    http://${WEBSERVER}"
echo "  🤖 포즈 서버:  http://${POSE_SERVER}"
echo "  🧠 MOSU 서버:  http://${MOSU_SERVER}"

echo -e "\n🔍 네트워크 연결 테스트:"

# 웹 서버 확인
echo -n "  웹 서버 (${WEBSERVER}): "
if curl -s --connect-timeout 5 "http://${WEBSERVER}/" > /dev/null; then
    echo "✅ 연결 성공"
else
    echo "❌ 연결 실패"
fi

# 포즈 서버 확인
echo -n "  포즈 서버 (${POSE_SERVER}): "
if curl -s --connect-timeout 5 "http://${POSE_SERVER}/health" > /dev/null; then
    echo "✅ 연결 성공"
else
    echo "❌ 연결 실패"
fi

# MOSU 서버 확인
echo -n "  MOSU 서버 (${MOSU_SERVER}): "
if curl -s --connect-timeout 5 "http://${MOSU_SERVER}/health" > /dev/null; then
    echo "✅ 연결 성공"
else
    echo "❌ 연결 실패"
fi

echo -e "\n🔧 포트 상태 확인:"
echo "  웹 서버 포트 8000:"
netstat -tuln | grep ":8000 " || echo "    ❌ 포트 8000 열려있지 않음"

echo "  포즈 서버 포트 5000:"
netstat -tuln | grep ":5000 " || echo "    ❌ 포트 5000 열려있지 않음"

echo "  MOSU 서버 포트 8002:"
netstat -tuln | grep ":8002 " || echo "    ❌ 포트 8002 열려있지 않음"

echo -e "\n📊 시스템 상태:"
echo "  현재 시간: $(date)"
echo "  현재 IP: $(hostname -I | awk '{print $1}')"

echo -e "\n💡 서버 시작 명령:"
echo "  웹 서버:   cd web-main && python -m uvicorn app.main:app --host 192.168.100.90 --port 8000"
echo "  포즈 서버: python pose-server.py --host 192.168.100.135 --port 5000"
echo "  MOSU 서버: cd mosu-server && ./start_real_server.sh"
