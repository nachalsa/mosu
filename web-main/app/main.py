from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from webcam.webcam_web import router as webcam_router
from video_process.video_web import router as video_result_router
from mosu_client import router as mosu_client_router

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(webcam_router, prefix="/webcam")
app.include_router(video_result_router, prefix="/video_result")
app.include_router(mosu_client_router, prefix="/mosu")

# 추후 파일 업로드 등 다른 라우터도 아래처럼 추가
# from upload.upload import router as upload_router
# app.include_router(upload_router, prefix="/upload")