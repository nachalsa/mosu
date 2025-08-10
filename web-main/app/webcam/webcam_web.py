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
        "image_count": webcam.image_count,
        "translating": webcam.realtime_translate, #webcam.recording,
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

@router.post("/start_realtime")
async def start_realtime():
    message = webcam.start_realtime()
    return {"message": message}

@router.post("/stop_realtime")
async def stop_realtime():
    message = webcam.stop_realtime()
    return {"message": message}

@router.post("/realtime_last_data")
async def realtime_last_data():
    return JSONResponse({
        "result": webcam.last_server_result or "결과 없음"
    })

@router.post("/test_auto_stop")
async def test_auto_stop():
    """자동 종료 테스트용 엔드포인트"""
    webcam.last_server_result = "미안함"
    return {"message": "자동 종료 테스트 트리거됨"}

@router.post("/rotate_camera")
async def rotate_camera():
    """카메라 화면 회전 (90도씩)"""
    current_angle = webcam.rotation_angle
    new_angle = (current_angle + 90) % 360
    message = webcam.set_rotation(new_angle)
    return {"message": message, "angle": new_angle}

@router.get("/get_rotation")
async def get_rotation():
    """현재 회전 각도 조회"""
    return {"angle": webcam.rotation_angle}

@router.post("/set_rotation")
async def set_rotation(request: dict):
    """특정 각도로 회전 설정"""
    angle = request.get('angle', 0)
    if angle not in [0, 270]:
        return {"message": "지원되지 않는 회전 각도입니다", "angle": webcam.rotation_angle}
    
    message = webcam.set_rotation(angle)
    return {"message": message, "angle": webcam.rotation_angle}