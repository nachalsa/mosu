# 수화 영상 처리 및 응답 결과 확인용 서버

- 라즈베리파이 구동용
- 웹캠 녹화/녹화종료

실행법
\mosu\web-main\app에서
uvicorn main:app --reload --host 0.0.0.0 --port 8000

실행 웹페이지
http://localhost:8000/static/webcam.html