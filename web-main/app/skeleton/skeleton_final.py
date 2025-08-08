import os
import json
import cv2

COCO_WHOLEBODY_SKELETON = [
    # Body (0~16)
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],

    # Left Hand (91~111)
    [91, 92], [92, 93], [93, 94], [94, 95],         # Thumb
    [91, 96], [96, 97], [97, 98], [98, 99],         # Index
    [91, 100], [100, 101], [101, 102], [102, 103],  # Middle
    [91, 104], [104, 105], [105, 106], [106, 107],  # Ring
    [91, 108], [108, 109], [109, 110], [110, 111],  # Pinky

    # Right Hand (112~132)
    [112, 113], [113, 114], [114, 115], [115, 116],      # Thumb
    [112, 117], [117, 118], [118, 119], [119, 120],      # Index
    [112, 121], [121, 122], [122, 123], [123, 124],      # Middle
    [112, 125], [125, 126], [126, 127], [127, 128],      # Ring
    [112, 129], [129, 130], [130, 131], [131, 132],      # Pinky
]

def draw_keypoints_wholebody(image, keypoints, scores, threshold=2.0):
    num_points = 133

    for idx in range(num_points):
        if 17 <= idx <= 22:
            continue  # 발 keypoint 무시
        if scores[idx] > threshold:
            x, y = keypoints[idx]
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

    for idx1, idx2 in COCO_WHOLEBODY_SKELETON:
        if 17 <= idx1 <= 22 or 17 <= idx2 <= 22:
            continue  # 발 keypoint 포함된 연결 무시
        if scores[idx1] > threshold and scores[idx2] > threshold:
            x1, y1 = keypoints[idx1]
            x2, y2 = keypoints[idx2]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

def get_latest_capture_dirs():
    """
    현재 파일 기준 captured_datas 폴더에서 가장 최근 폴더 경로 반환
    """
    skeleton_dir = os.path.dirname(__file__)
    captured_datas_dir = os.path.join(skeleton_dir, '..', 'captured_datas')

    all_dirs = [d for d in os.listdir(captured_datas_dir)
                if os.path.isdir(os.path.join(captured_datas_dir, d)) and d.isdigit()]

    if not all_dirs:
        raise RuntimeError("⚠️ 'captured_datas' 폴더에 유효한 캡처 폴더가 없습니다.")

    latest_dir_name = max(all_dirs)
    original_img_dir = os.path.join(captured_datas_dir, latest_dir_name)
    cropped_data_dir = os.path.join(captured_datas_dir, f"{latest_dir_name}-crop")

    return latest_dir_name, original_img_dir, cropped_data_dir

def visualize_wholebody_on_original(threshold=2.0):
    latest_dir_name, original_img_dir, cropped_data_dir = get_latest_capture_dirs()

    # output 디렉토리 이름을 output_시간형태로
    output_dir = os.path.join(os.path.dirname(__file__), f'output_{latest_dir_name}')
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(cropped_data_dir):
        if fname.endswith('.json') and 'pose' in fname:
            json_path = os.path.join(cropped_data_dir, fname)

            with open(json_path, 'r') as f:
                data = json.load(f)

            base_name = '_'.join(fname.split('_')[:2]) + '.jpg'
            original_img_path = os.path.join(original_img_dir, base_name)

            if not os.path.exists(original_img_path):
                print(f"⚠️ 원본 이미지 없음: {original_img_path}")
                continue

            img = cv2.imread(original_img_path)
            if img is None:
                print(f"⚠️ 이미지 읽기 실패: {original_img_path}")
                continue

            if "original_keypoints" not in data or "scores" not in data:
                print(f"⚠️ 'original_keypoints' 또는 'scores' 누락: {fname}")
                continue

            keypoints = data["original_keypoints"]
            scores = data["scores"]

            if len(keypoints) != 133 or len(scores) != 133:
                print(f"⚠️ 잘못된 길이: {fname}")
                continue

            draw_keypoints_wholebody(img, keypoints, scores, threshold=threshold)

            out_path = os.path.join(output_dir, base_name)
            cv2.imwrite(out_path, img)
            print(f"✅ 저장 완료: {out_path}")

if __name__ == "__main__":
    visualize_wholebody_on_original()
