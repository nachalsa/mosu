#!/usr/bin/env python3
"""
로컬 MOSU 서버 (테스트용)
- 로컬 포즈서버 (localhost:5001) 사용
- 로컬 웹 인터페이스
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import MosuServer
from network_pose_estimator import NetworkPoseEstimator
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalMosuServer(MosuServer):
    """로컬 테스트용 MOSU 서버"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        # 로컬 설정으로 부모 초기화
        super().__init__(
            model_path=model_path,
            device=device,
            host="localhost",
            port=8002
        )
        
        # 로컬 포즈 추정기로 교체
        try:
            self.pose_estimator = NetworkPoseEstimator("http://localhost:5001")
            logger.info("🏠 로컬 네트워크 포즈 추정기 사용")
        except Exception as e:
            logger.warning(f"⚠️ 로컬 네트워크 포즈 추정기 실패: {e}")
    
    def get_index_html(self):
        """로컬 테스트용 HTML (localhost 연결)"""
        return """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MOSU 로컬 테스트</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    min-height: 100vh;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                h1 {
                    text-align: center;
                    color: white;
                    font-size: 2.5em;
                    margin-bottom: 30px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }
                
                .test-banner {
                    background: rgba(255, 193, 7, 0.2);
                    border: 2px solid #ffc107;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                    font-weight: bold;
                }
                
                .main-content {
                    display: grid;
                    grid-template-columns: 2fr 1fr;
                    gap: 30px;
                    margin-bottom: 30px;
                }
                
                .video-section, .controls-section {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 25px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                }
                
                #video {
                    width: 100%;
                    height: 360px;
                    background-color: #222;
                    border-radius: 10px;
                    object-fit: cover;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }
                
                button {
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    margin: 5px;
                    border-radius: 25px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 500;
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                }
                
                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                }
                
                button:disabled {
                    background: #666;
                    cursor: not-allowed;
                    transform: none;
                }
                
                .status {
                    padding: 15px;
                    border-radius: 10px;
                    margin: 10px 0;
                    font-weight: 500;
                }
                
                .status.connected {
                    background: rgba(76, 175, 80, 0.2);
                    border: 2px solid #4CAF50;
                }
                
                .status.disconnected {
                    background: rgba(244, 67, 54, 0.2);
                    border: 2px solid #f44336;
                }
                
                .results-section {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 25px;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    grid-column: span 2;
                }
                
                .current-word {
                    font-size: 3em;
                    font-weight: bold;
                    text-align: center;
                    margin: 20px 0;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 15px;
                    min-height: 80px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏠 MOSU 로컬 테스트</h1>
                
                <div class="test-banner">
                    ⚠️ 로컬 테스트 모드 - 모든 서비스가 localhost에서 실행됩니다
                </div>
                
                <div class="main-content">
                    <div class="video-section">
                        <h3>📹 웹캠 영상</h3>
                        <video id="video" autoplay muted playsinline></video>
                        <div>
                            <button id="start-btn" onclick="startCamera()">🎥 카메라 시작</button>
                            <button id="stop-btn" onclick="stopCamera()" disabled>⏹️ 카메라 정지</button>
                            <button onclick="clearResults()">🧹 결과 지우기</button>
                        </div>
                    </div>
                    
                    <div class="controls-section">
                        <h3>🔗 연결 상태</h3>
                        <div id="connection-status" class="status disconnected">
                            ❌ 연결 안됨
                        </div>
                        
                        <h3>⚙️ 로컬 설정</h3>
                        <p>MOSU 서버: localhost:8002</p>
                        <p>포즈 서버: localhost:5001</p>
                        <p>웹 서버: localhost:8000</p>
                    </div>
                </div>
                
                <div class="results-section">
                    <h3>🎯 인식 결과</h3>
                    <div class="current-word" id="current-word">대기 중...</div>
                    <div style="text-align: center;">
                        <span>신뢰도: </span>
                        <span id="confidence" style="background: rgba(33, 150, 243, 0.2); color: #2196F3; padding: 4px 8px; border-radius: 5px;">0%</span>
                    </div>
                </div>
            </div>

            <script>
                let video = document.getElementById('video');
                let ws = null;
                let isStreaming = false;

                // WebSocket 연결 (localhost)
                function connectWebSocket() {
                    const wsUrl = 'ws://localhost:8002/ws';
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        const statusEl = document.getElementById('connection-status');
                        statusEl.textContent = '✅ 로컬 연결됨';
                        statusEl.className = 'status connected';
                        console.log('로컬 MOSU 서버 연결 성공');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.type === 'result' && data.word) {
                            document.getElementById('current-word').textContent = data.word;
                            document.getElementById('confidence').textContent = 
                                Math.round(data.confidence * 100) + '%';
                        }
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocket 오류:', error);
                        const statusEl = document.getElementById('connection-status');
                        statusEl.textContent = '❌ 연결 오류';
                        statusEl.className = 'status disconnected';
                    };
                    
                    ws.onclose = function() {
                        const statusEl = document.getElementById('connection-status');
                        statusEl.textContent = '❌ 연결 끊김';
                        statusEl.className = 'status disconnected';
                        
                        // 자동 재연결 시도
                        setTimeout(connectWebSocket, 3000);
                    };
                }

                // 카메라 시작
                async function startCamera() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({
                            video: { width: 640, height: 480, frameRate: 15 }
                        });
                        
                        video.srcObject = stream;
                        isStreaming = true;
                        
                        document.getElementById('start-btn').disabled = true;
                        document.getElementById('stop-btn').disabled = false;
                        
                        startFrameCapture();
                        
                    } catch (err) {
                        console.error('카메라 접근 실패:', err);
                        alert('카메라에 접근할 수 없습니다: ' + err.message);
                    }
                }

                // 카메라 정지
                function stopCamera() {
                    if (video.srcObject) {
                        video.srcObject.getTracks().forEach(track => track.stop());
                        video.srcObject = null;
                    }
                    
                    isStreaming = false;
                    document.getElementById('start-btn').disabled = false;
                    document.getElementById('stop-btn').disabled = true;
                }

                // 프레임 캡처 및 전송
                function startFrameCapture() {
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    function captureFrame() {
                        if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) {
                            return;
                        }
                        
                        canvas.width = video.videoWidth || 640;
                        canvas.height = video.videoHeight || 480;
                        
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        
                        canvas.toBlob(function(blob) {
                            if (blob && ws.readyState === WebSocket.OPEN) {
                                blob.arrayBuffer().then(buffer => {
                                    const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
                                    
                                    ws.send(JSON.stringify({
                                        type: 'frame',
                                        data: base64,
                                        timestamp: Date.now()
                                    }));
                                });
                            }
                        }, 'image/jpeg', 0.8);
                        
                        setTimeout(captureFrame, 1000/15);
                    }
                    
                    captureFrame();
                }

                // 결과 지우기
                function clearResults() {
                    document.getElementById('current-word').textContent = '대기 중...';
                    document.getElementById('confidence').textContent = '0%';
                }

                // 페이지 로드 시 연결
                window.onload = function() {
                    connectWebSocket();
                };

                // 페이지 종료 시 정리
                window.onbeforeunload = function() {
                    stopCamera();
                    if (ws) ws.close();
                };
            </script>
        </body>
        </html>
        """

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MOSU Local Test Server")
    parser.add_argument("--model", type=str, 
                       default="../mosumodel/best_model_stage_1.pt",
                       help="수화 인식 모델 경로")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "xpu"],
                       help="추론 디바이스")
    
    args = parser.parse_args()
    
    from pathlib import Path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    try:
        print("🏠 MOSU 로컬 테스트 서버 시작")
        print("   - 주소: http://localhost:8002")
        print("   - 포즈서버: localhost:5001 필요")
        print("   - Ctrl+C로 종료")
        
        server = LocalMosuServer(
            model_path=str(model_path),
            device=args.device
        )
        
        server.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ 로컬 서버 종료")
    except Exception as e:
        print(f"❌ 로컬 서버 실행 실패: {e}")

if __name__ == "__main__":
    main()
