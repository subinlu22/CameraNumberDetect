# -*- coding: utf-8 -*-
"""
전처리(extract_digit_patch)가 숫자를 제대로 뽑아내고 있는지
눈으로 직접 확인하기 위한 디버그용 스크립트.

train_digit_cnn.py랑 완전히 똑같은 전처리 로직을 그대로 가져와서,
각 폴더(화이트_1, 파랑_2 등)에서 몇 장씩 샘플을 뽑아 처리한 다음
"원본 | 처리결과" 나란히 붙인 이미지를 만들어서 저장함.

■ 실행 방법
    C:\\Users\\dev\\venv311\\Scripts\\python.exe visualize_patches.py

■ 결과물
    patch_check.png - 이 파일 열어서 오른쪽(처리결과)에 숫자 모양이
    깨끗하게 하얗게 나오는지, 아니면 노이즈/글리터 반사까지 같이
    묻어나오는지 확인하면 됨
"""

import os
import cv2
import numpy as np

TRAINING_DATA_DIR = r"C:\Users\dev\CameraProject\CameraNumberDetect\TrainingData"
IMAGE_SIZE = 64
SAMPLES_PER_FOLDER = 3  # 폴더마다 몇 장씩 확인해볼지


def imread_korean_path(file_path):
    """한글 경로도 읽을 수 있는 imread (train_digit_cnn.py와 동일 방식)"""
    file_bytes = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def extract_digit_patch(image_bgr, out_size=IMAGE_SIZE):
    """train_digit_cnn.py의 extract_digit_patch와 완전히 동일한 로직"""
    img = image_bgr.astype(np.float32)
    h, w = img.shape[:2]

    # [수정] 사각형 귀퉁이(뚜껑 바깥 배경)를 원형 ROI로 아예 분석에서 제외
    # (train_digit_cnn.py와 동일 로직 - 자세한 이유는 그쪽 주석 참고)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_radius = int(min(h, w) * 0.48)
    cv2.circle(roi_mask, (w // 2, h // 2), roi_radius, 255, -1)

    pixels_in_roi = img[roi_mask == 255]
    bg_color = np.median(pixels_in_roi, axis=0)

    diff = np.linalg.norm(img - bg_color, axis=2)
    diff[roi_mask == 0] = 0

    diff_normalized = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)
    diff_normalized = cv2.GaussianBlur(diff_normalized, (5, 5), 0)

    _, mask = cv2.threshold(diff_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[roi_mask == 0] = 0

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(biggest)
    if box_w == 0 or box_h == 0:
        return None

    digit_crop = mask[y:y + box_h, x:x + box_w]
    side = max(box_w, box_h)
    padded = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - box_h) // 2
    x_offset = (side - box_w) // 2
    padded[y_offset:y_offset + box_h, x_offset:x_offset + box_w] = digit_crop

    resized = cv2.resize(padded, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized


def main():
    rows = []

    for folder_name in sorted(os.listdir(TRAINING_DATA_DIR)):
        folder_path = os.path.join(TRAINING_DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
        files = files[:SAMPLES_PER_FOLDER]

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            image = imread_korean_path(file_path)
            if image is None:
                continue

            patch = extract_digit_patch(image)
            if patch is None:
                # 숫자 못 찾은 경우도 빨간 X 표시로 보여줌
                patch = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                cv2.line(patch, (0, 0), (IMAGE_SIZE, IMAGE_SIZE), 255, 3)
                cv2.line(patch, (0, IMAGE_SIZE), (IMAGE_SIZE, 0), 255, 3)

            # [원본] 크기 통일해서 옆에 나란히 붙이기 위해 리사이즈
            original_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

            # 원본 | 처리결과 나란히 붙이기
            pair = np.hstack([original_resized, patch_bgr])

            # 폴더명 라벨 붙이기 (위에 검은 띠 + 흰 글씨)
            label_strip = np.zeros((25, pair.shape[1], 3), dtype=np.uint8)
            cv2.putText(label_strip, folder_name, (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            labeled_pair = np.vstack([label_strip, pair])

            rows.append(labeled_pair)

    if not rows:
        print("[에러] 샘플을 하나도 못 만들었어. TRAINING_DATA_DIR 경로 확인해줘.")
        return

    # 세로로 전부 이어 붙여서 하나의 큰 이미지로 저장
    montage = np.vstack(rows)
    cv2.imwrite("patch_check.png", montage)
    print(f"patch_check.png 저장 완료! (총 {len(rows)}개 샘플)")
    print("이 파일 열어서 왼쪽(원본)이랑 오른쪽(처리결과) 비교해서 확인해줘.")


if __name__ == "__main__":
    main()
