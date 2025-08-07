import os
import json
import cv2

# COCO WholeBody skeleton (발 제외)
COCO_WHOLEBODY_SKELETON = [
    # Body (0~16)
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],

    # Face (23~90) 일부 윤곽선만
    [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],

    # Left Hand (91~111)
    [91, 92], [92, 93], [93, 94],
    [91, 95], [95, 96], [96, 97],
    [91, 98], [98, 99], [99, 100],
    [91, 101], [101, 102], [102, 103],
    [91, 104], [104, 105], [105, 106],

    # Right Hand (112~132)
    [112, 113], [113, 114], [114, 115],
    [112, 116], [116, 117], [117, 118],
    [112, 119], [119, 120], [120, 121],
    [112, 122], [122, 123], [123, 124],
    [112, 125], [125, 126], [126, 127],
]

def draw_keypoints_wholebody_v2(image, keypoints, scores, threshold=9.5):
    """
    image: 원본 BGR 이미지
    keypoints: [[x,y], ...] 133개
    scores: [score, ...] 133개
    threshold: 스코어 임계값 미만 keypoint 무시
    """
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

def visualize_wholebody_on_original(
    original_img_dir='/home/js/mosu/mosu/web-main/app/captured_datas/20250807210309',
    cropped_data_dir='/home/js/mosu/mosu/web-main/app/captured_datas/20250807210309-crop',
    output_dir='/home/js/mosu/mosu/web-main/app/skeleton/output',
    threshold=0.3
):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(cropped_data_dir):
        if fname.endswith('.json') and 'pose' in fname:
            json_path = os.path.join(cropped_data_dir, fname)

            with open(json_path, 'r') as f:
                data = json.load(f)

            # img_0001_1_pose_cebb6b13.json -> img_0001.jpg
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

            draw_keypoints_wholebody_v2(img, keypoints, scores, threshold=threshold)

            out_path = os.path.join(output_dir, base_name)
            cv2.imwrite(out_path, img)
            print(f"✅ 저장 완료: {out_path}")

if __name__ == "__main__":
    visualize_wholebody_on_original()
