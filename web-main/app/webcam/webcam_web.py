from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from .webcam import WebcamCapture
import traceback
import logging

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
    message = webcam.start_capture_images() #webcam.start_recording()
    return {"message": message}

@router.post("/stop_recording")
async def stop_recording():
    message = webcam.stop_capture_images() # webcam.stop_recording()
    return {"message": message}

@router.get("/status")
async def get_status():
    return {
        "camera_initialized": webcam.cap is not None and webcam.cap.isOpened(),
        "recording": webcam.capturing_images, #webcam.recording,
        "video_count": webcam.video_count,
        "image_count": webcam.image_count
    }

@router.post("/send_to_server")
async def send_to_server():
    try:
        result = webcam.process_latest_folder_with_yolo()
        return JSONResponse({"message": result})
    except Exception as e:
                # 콘솔에 전체 에러 스택 출력
        traceback.print_exc()
        logging.error(f"send_to_server error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)