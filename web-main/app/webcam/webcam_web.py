from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from .webcam import WebcamCapture

router = APIRouter()
webcam = WebcamCapture()

@router.on_event("startup")
async def startup_event():
    webcam.initialize_camera()

@router.on_event("shutdown")
async def shutdown_event():
    webcam.release_camera()

@router.get("/")
async def root():
    return {"message": "FastAPI Webcam Capture Server"}

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        webcam.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/capture_image")
async def capture_image():
    image_path = webcam.save_image()
    if image_path:
        return {"message": f"이미지 저장됨: {image_path}"}
    return {"error": "이미지 캡처 실패"}

@router.post("/start_recording")
async def start_recording():
    message = webcam.start_recording()
    return {"message": message}

@router.post("/stop_recording")
async def stop_recording():
    message = webcam.stop_recording()
    return {"message": message}

@router.get("/status")
async def get_status():
    return {
        "camera_initialized": webcam.cap is not None and webcam.cap.isOpened(),
        "recording": webcam.recording,
        "video_count": webcam.video_count,
        "image_count": webcam.image_count
    }