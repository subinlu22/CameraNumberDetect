# -*- coding: utf-8 -*-
"""
학습에 전혀 안 쓰인 새 사진들로 모델이 진짜로 잘 맞히는지 확인하는 스크립트.

(검증 정확도 100%가 나왔는데, 데이터가 적어서 학습용/검증용에 거의
 똑같은 사진이 우연히 섞여 들어갔을 가능성이 있어 - 완전히 새로 찍은
 사진으로 다시 확인해서 진짜 실력인지 검증하는 용도)

■ 준비물
    TestData 폴더를 새로 만들고, 그 안에 TrainingData랑 똑같은 구조로
    (화이트_1, 파랑_2 등) 새로 찍은 사진들을 넣어둘 것

■ 실행 방법
    C:\\Users\\dev\\venv311\\Scripts\\python.exe -m pip install onnxruntime --break-system-packages
    C:\\Users\\dev\\venv311\\Scripts\\python.exe test_model.py
"""

import os
import json

import cv2
import numpy as np
import onnxruntime as ort

TEST_DATA_DIR = r"C:\Users\dev\CameraProject\CameraNumberDetect\TestData"
IMAGE_SIZE = 64


def imread_korean_path(file_path):
    file_bytes = np.fromfile(file_path, dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def extract_digit_patch(image_bgr, out_size=IMAGE_SIZE):
    """train_digit_cnn.py의 전처리 로직과 완전히 동일 (반드시 똑같이 유지해야 함 -
    학습 때와 다른 방식으로 전처리하면 모델이 처음 보는 형태의 입력이 되어버림)"""
    img = image_bgr.astype(np.float32)
    h, w = img.shape[:2]

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
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

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
    with open("label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)  # {"0":"1", "1":"2", "2":"3"}

    session = ort.InferenceSession("digit_cnn.onnx")

    total = 0
    correct = 0
    wrong_cases = []

    for folder_name in sorted(os.listdir(TEST_DATA_DIR)):
        folder_path = os.path.join(TEST_DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parts = folder_name.split("_")
        if len(parts) != 2:
            continue
        true_digit = parts[1]  # 폴더명에서 정답 숫자 추출

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".png"):
                continue

            file_path = os.path.join(folder_path, file_name)
            image = imread_korean_path(file_path)
            if image is None:
                continue

            patch = extract_digit_patch(image)
            if patch is None:
                print(f"[숫자 못 찾음] {folder_name}/{file_name}")
                continue

            # [ONNX 입력 형태로 변환] (1, 1, 64, 64) 배치+채널+높이+너비 순서, 0~1로 정규화
            input_tensor = patch.astype(np.float32) / 255.0
            input_tensor = input_tensor[np.newaxis, np.newaxis, :, :]

            outputs = session.run(None, {"input": input_tensor})
            predicted_index = int(np.argmax(outputs[0]))
            predicted_digit = label_map[str(predicted_index)]

            total += 1
            is_correct = (predicted_digit == true_digit)
            if is_correct:
                correct += 1
            else:
                wrong_cases.append((folder_name, file_name, true_digit, predicted_digit))

            mark = "O" if is_correct else "X"
            print(f"[{mark}] {folder_name}/{file_name} → 정답:{true_digit}, 예측:{predicted_digit}")

    if total == 0:
        print("[에러] TestData 폴더에서 테스트할 사진을 하나도 못 찾았어. 경로/폴더구조 확인해줘.")
        return

    accuracy = correct / total * 100
    print(f"\n=== 최종 결과: {correct}/{total} 정답 ({accuracy:.1f}%) ===")

    if wrong_cases:
        print("\n틀린 케이스들:")
        for folder, file, true_d, pred_d in wrong_cases:
            print(f"  {folder}/{file} - 정답:{true_d}인데 {pred_d}로 예측함")


if __name__ == "__main__":
    main()
