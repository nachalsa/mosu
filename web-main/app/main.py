from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from webcam.webcam_web import router as webcam_router

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(webcam_router, prefix="/webcam")

# 추후 파일 업로드 등 다른 라우터도 아래처럼 추가
# from upload.upload import router as upload_router
# app.include_router(upload_router, prefix="/upload")