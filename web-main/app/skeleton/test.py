import os
import cv2
import json

# 경로 설정
original_dir = '/home/js/mosu/mosu/web-main/app/captured_datas/20250807184000'   # 원본 영상 프레임 디렉토리
json_dir = '/home/js/mosu/mosu/web-main/app/captured_datas/20250807184000-crop'                 # keypoints + bbox 포함된 json들
output_dir = '/home/js/mosu/mosu/web-main/app/skeleton/output'              # 시각화 이미지 저장 디렉토리
os.makedirs(output_dir, exist_ok=True)

# resize된 크롭 이미지 사이즈
crop_width = 288
crop_height = 384

# COCO 스타일 연결 순서 (원하면 수정 가능)
skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (0, 16),
    (14, 19), (19, 20),
    (14, 21), (11, 22)
]

def map_keypoints_to_original(keypoints, bbox, input_size=(288, 384)):
    x1, y1, x2, y2 = bbox
    scale_x = (x2 - x1) / input_size[0]
    scale_y = (y2 - y1) / input_size[1]

    mapped = []
    for kp in keypoints:
        if len(kp) == 2:
            x, y = kp
            v = 2
        elif len(kp) == 3:
            x, y, v = kp
        else:
            continue

        if v > 0:
            orig_x = x1 + x * scale_x
            orig_y = y1 + y * scale_y
            mapped.append((int(orig_x), int(orig_y), v))
    return mapped

# keypoint, skeleton 색상
kpt_color = (0, 255, 0)
line_color = (255, 0, 0)

# 원본 이미지 리스트 가져오기
image_names = sorted([f for f in os.listdir(original_dir) if f.endswith('.jpg')])

for img_name in image_names:
    base_name = os.path.splitext(img_name)[0]  # ex: img_0001
    original_path = os.path.join(original_dir, img_name)
    image = cv2.imread(original_path)

    # 현재 프레임에 해당하는 모든 json 가져오기
    person_jsons = [f for f in os.listdir(json_dir) if f.startswith(base_name) and f.endswith('.json')]

    for json_file in person_jsons:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        bbox = data.get('bbox', [])
        keypoints = data.get('keypoints', [])

        if not bbox or not keypoints:
            print(f"Invalid data in {json_file}")
            continue

        # keypoints 복원
        keypoints_orig = map_keypoints_to_original(keypoints, bbox)

        # keypoint 그리기
        for x, y, v in keypoints_orig:
            if v > 0:
                cv2.circle(image, (x, y), 3, kpt_color, -1)

        # skeleton 연결
        for i, j in skeleton:
            if i < len(keypoints_orig) and j < len(keypoints_orig):
                xi, yi, vi = keypoints_orig[i]
                xj, yj, vj = keypoints_orig[j]
                if vi > 0 and vj > 0:
                    cv2.line(image, (xi, yi), (xj, yj), line_color, 2)

    # 저장
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")
