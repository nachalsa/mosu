#!/usr/bin/env python3
"""
MOSU 서버 테스트 (최소 버전)
"""

print("🤖 MOSU 서버 시작...")

try:
    import sys
    import os
    from pathlib import Path
    
    print(f"✅ Python: {sys.version}")
    print(f"✅ 작업 디렉토리: {os.getcwd()}")
    
    # 기본 패키지들 확인
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        print("✅ FastAPI 사용 가능")
    except ImportError as e:
        print(f"❌ FastAPI 없음: {e}")
        print("pip install fastapi uvicorn 실행 필요")
        exit(1)
    
    try:
        import uvicorn
        print("✅ Uvicorn 사용 가능")
    except ImportError as e:
        print(f"❌ Uvicorn 없음: {e}")
        exit(1)
    
    try:
        import numpy as np
        print("✅ NumPy 사용 가능")
    except ImportError as e:
        print(f"❌ NumPy 없음: {e}")
        print("pip install numpy 실행 필요")
        exit(1)
    
    # 모델 파일 확인
    model_path = Path("../mosumodel/best_model_stage_1.pt")
    if model_path.exists():
        print(f"✅ 모델 파일 존재: {model_path}")
    else:
        print(f"⚠️  모델 파일 없음: {model_path} (더미 모드로 실행)")
    
    # 간단한 웹 서버 생성
    app = FastAPI(title="MOSU 테스트 서버")
    
    @app.get("/")
    def root():
        return {
            "message": "MOSU 서버가 실행 중입니다!",
            "status": "healthy",
            "endpoints": [
                "/",
                "/health", 
                "/test"
            ]
        }
    
    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_available": model_path.exists(),
            "timestamp": __import__('time').time()
        }
    
    @app.get("/test", response_class=HTMLResponse)
    def test_page():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MOSU 서버 테스트</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>🤖 MOSU 서버 테스트 페이지</h1>
            <p>서버가 정상적으로 실행되고 있습니다!</p>
            <ul>
                <li><a href="/">홈</a></li>
                <li><a href="/health">상태 확인</a></li>
            </ul>
        </body>
        </html>
        """
    
    print("\n🚀 서버 시작...")
    print("   - 주소: http://localhost:8001")
    print("   - 테스트: http://localhost:8001/test")
    print("   - Ctrl+C로 종료")
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
