from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
import os
import glob
import time
import cv2
from fastapi.responses import FileResponse
from fastapi import Query

router = APIRouter()

def get_latest_capture_folder():
    base_dir = "captured_datas"
    if not os.path.exists(base_dir):
        return None
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not folders:
        return None
    latest_folder = max(folders, key=os.path.getmtime)
    return latest_folder

def iter_latest_images():
    folder = get_latest_capture_folder()
    if not folder:
        return
    images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                )
        time.sleep(0.1)

@router.get("/video_result_feed")
async def video_result_feed():
    return StreamingResponse(
        iter_latest_images(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video_result_word")
async def video_result_word(filename: str = Query(None)):
    # filename이 주어지면 해당 파일명에서 단어 추출
    if filename:
        word = filename.split('.')[0]
        return JSONResponse({"word": word})
    # 없으면 기존 방식
    folder = get_latest_capture_folder()
    if not folder:
        return JSONResponse({"word": ""})
    images = sorted(glob.glob(os.path.join(folder, "*.jpg"))) # 수정필요.
    if not images:
        return JSONResponse({"word": ""})
    word = os.path.basename(images[-1]).split('.')[0]
    return JSONResponse({"word": word})


@router.get("/video_result_images")
async def video_result_images():
    folder = get_latest_capture_folder()
    if not folder:
        return JSONResponse({"images": []})
    images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    # 이미지 파일명만 반환
    image_urls = [f"/video_result/image/{os.path.basename(img)}" for img in images]
    return JSONResponse({"images": image_urls})

@router.get("/image/{filename}")
async def get_image(filename: str):
    folder = get_latest_capture_folder()
    if not folder:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    img_path = os.path.join(folder, filename)
    if not os.path.exists(img_path):
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    return FileResponse(img_path, media_type="image/jpeg")