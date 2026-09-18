# -*- coding: utf-8 -*-
"""
숫자 모양(1/2/3) 분류 CNN 학습 스크립트

■ 전체 흐름
1. TrainingData 폴더(화이트_1, 파랑_2, 빨강_3 등 9개 폴더) 안의 사진들을 읽음
2. 폴더 이름에서 색깔은 무시하고, 뒤에 붙은 숫자(1/2/3)만 라벨로 사용
   → 예: "화이트_1", "파랑_1", "빨강_1" → 전부 "숫자 1" 클래스로 합쳐짐
3. 각 사진에서 "배경색과 다른 부분"만 뽑아내서 숫자 모양 흑백 마스크로 변환
   (색깔 값 자체를 안 쓰고 "배경이랑 얼마나 다른가"만 보니까, 뚜껑이 무슨 색이든 상관없음)
4. 마스크에서 숫자 부분만 딱 잘라서 64x64 정사각형으로 통일
5. 회전/이동/확대축소 랜덤 변형(augmentation)으로 데이터를 부풀려서 학습
6. 간단한 CNN으로 학습 (1/2/3 세 가지 중 하나로 분류)
7. 학습 끝나면 .onnx 파일로 저장 (나중에 C#에서 ONNX Runtime으로 불러다 씀)

■ 실행 전 준비 (학교 PC, venv311 기준)
    pip install torch torchvision opencv-python numpy scikit-learn --break-system-packages
    (RTX 4070 있으니까 GPU용 torch 설치되면 훨씬 빠름 - 이미 설치돼있으면 그냥 넘어가도 됨)

■ 실행 방법
    python train_digit_cnn.py

■ 결과물
    digit_cnn.onnx      → C#에서 불러다 쓸 모델 파일
    digit_cnn_best.pth  → 파이토치용 체크포인트 (혹시 나중에 다시 학습 이어할 때 필요)
    label_map.json       → 클래스 번호(0,1,2)가 실제 숫자(1,2,3) 중 뭔지 매핑 정보
"""

import os
import json
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# =============================================================
# [설정값] - 필요하면 여기 숫자들만 조절하면 됨
# =============================================================
TRAINING_DATA_DIR = r"C:\Users\dev\CameraProject\CameraNumberDetect\TrainingData"  # 사진 폴더 위치
IMAGE_SIZE = 64          # CNN에 넣을 최종 이미지 크기 (64x64 정사각형)
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
VAL_RATIO = 0.2          # 전체 데이터 중 20%는 검증용으로 따로 뺴둠 (학습에 안 씀, 실력 확인용)
RANDOM_SEED = 42

# 재현 가능하게 랜덤 시드 고정 (매번 똑같은 조건으로 비교하려고)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =============================================================
# [전처리 함수] 사진 한 장 → 숫자 모양만 담긴 64x64 흑백 마스크로 변환
# =============================================================
def extract_digit_patch(image_bgr, out_size=IMAGE_SIZE):
    """
    뚜껑이 크롭된 컬러 사진을 받아서, 배경(뚜껑)과 다른 부분(=숫자)만
    뽑아낸 정사각형 흑백 이미지로 바꿔주는 함수.

    색깔 값 자체(Hue 등)를 안 쓰고 "배경이랑 얼마나 다른 색인가"라는
    상대적인 기준만 쓰기 때문에, 화이트/파랑/빨강 뚜껑 전부 똑같은
    코드로 처리 가능함 (뚜껑 색깔별로 조건 분기 안 해도 됨).
    """
    img = image_bgr.astype(np.float32)
    h, w = img.shape[:2]

    # [원형 관심영역(ROI)] 뚜껑은 원형인데 크롭은 사각형이라, 사각형의 네 귀퉁이는
    # 사실 뚜껑이 아니라 뚜껑 "바깥"(책상 배경)임.
    # [처음 버전의 문제] 화이트 뚜껑은 우연히 책상 배경이랑 밝기가 비슷해서 어쩌다 잘 됐지만,
    # 파랑/빨강처럼 색이 뚜렷한 뚜껑은 "뚜껑 전체"가 귀퉁이(책상)랑 크게 달라 보여서
    # 숫자 대신 뚜껑 전체가 통째로 "다른 부분"으로 잡혀버리는 문제가 있었음.
    # [해결] 귀퉁이를 아예 분석 대상에서 빼버림 - 이미지 가운데를 중심으로 한
    # 원형 영역(뚜껑이 있을 만한 자리)만 보고, 그 안에서만 배경색/숫자 판단을 함
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_radius = int(min(h, w) * 0.48)  # 살짝 여유를 둬서 원 경계 노이즈 방지
    cv2.circle(roi_mask, (w // 2, h // 2), roi_radius, 255, -1)

    # [배경색 추정] 원형 영역 안 픽셀들의 중간값(median)을 배경(=뚜껑 표면) 색으로 삼음
    # 숫자는 원 안에서도 일부 면적일 뿐이라, 중간값을 쓰면 뚜껑이 무슨 색이든
    # 상관없이 "뚜껑 자체의 색"이 안정적으로 나옴
    pixels_in_roi = img[roi_mask == 255]
    bg_color = np.median(pixels_in_roi, axis=0)

    # [신규] 색 거리 계산 "전에" 먼저 블러 처리 - 글리터 반짝임 노이즈를 미리 주변
    # 픽셀과 섞어서 완화함. 아래에 있던 블러(diff_normalized에 적용하는 것)는
    # 반짝이는 픽셀이 이미 극단적인 값을 만들어놓은 "뒤"라서 이미 늦은 시점이었음
    # ⚠️ C# 쪽 ExtractDigitPatchForCnn에도 커널 크기(7,7) 반드시 동일하게 적용됨 -
    #    학습(파이썬)과 추론(C#) 전처리가 한 글자도 다르면 안 되니 커널 크기 맞출 것
    blurred_for_diff = cv2.GaussianBlur(img, (7, 7), 0)

    # [배경과의 색 거리 계산] 각 픽셀이 배경색이랑 얼마나 다른지(유클리드 거리)
    # (bg_color는 원본 img 기준 median 그대로 사용 - 배경색 추정 자체는 안 건드림,
    #  거리 계산에 쓰는 이미지만 블러된 버전으로 교체)
    diff = np.linalg.norm(blurred_for_diff - bg_color, axis=2)
    diff[roi_mask == 0] = 0  # 원 바깥(귀퉁이)은 강제로 "배경"(0) 취급 - 절대 숫자로 안 잡히게

    diff_normalized = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)

    # [가우시안 블러] 글리터(반짝이) 재질 때문에 배경 표면 곳곳에 생기는
    # 작은 반사 스파클 노이즈를 흐릿하게 뭉개서 줄여줌
    diff_normalized = cv2.GaussianBlur(diff_normalized, (5, 5), 0)

    # [이진화] Otsu 자동 임계값으로 "많이 다른 부분"만 흰색(숫자)으로 분리
    _, mask = cv2.threshold(diff_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[roi_mask == 0] = 0  # 혹시 모르니 한 번 더 확실하게 원 바깥 차단

    # [모폴로지 열림(Opening)] 블러로도 못 지운 자잘한 점 노이즈들을 먼저 지움
    # (침식으로 작은 점들을 아예 없앤 다음, 팽창으로 원래 크기를 복원 -
    #  숫자처럼 큰 덩어리는 침식 후에도 살아남지만, 작은 점 노이즈는 침식 단계에서 사라짐)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # [모폴로지 닫힘] 글리터 반짝임 때문에 쪼개진 부분들을 뭉쳐줌
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # [제일 큰 윤곽선 찾기] = 숫자 영역
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(biggest)

    if box_w == 0 or box_h == 0:
        return None

    digit_crop = mask[y:y + box_h, x:x + box_w]

    # [정사각형으로 패딩] 숫자 모양의 가로세로 비율이 찌그러지지 않게,
    # 긴 쪽 기준으로 정사각형 검은 배경을 만들고 그 가운데에 숫자를 넣음
    side = max(box_w, box_h)
    padded = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - box_h) // 2
    x_offset = (side - box_w) // 2
    padded[y_offset:y_offset + box_h, x_offset:x_offset + box_w] = digit_crop

    # [최종 크기로 통일] 64x64로 리사이즈 - CNN 입력 크기는 항상 고정이어야 하니까
    resized = cv2.resize(padded, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized


# =============================================================
# [데이터 수집] TrainingData 폴더를 훑으면서 (이미지, 숫자라벨) 쌍 만들기
# =============================================================
def load_dataset(root_dir):
    patches = []
    labels = []
    skipped = 0

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # [폴더명 파싱] "화이트_1" → 색깔은 버리고 "1"만 라벨로 사용
        parts = folder_name.split("_")
        if len(parts) != 2:
            print(f"[건너뜀] 폴더명 형식이 이상함: {folder_name}")
            continue
        digit_label = parts[1]  # "1", "2", "3"

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(".png"):
                continue
            file_path = os.path.join(folder_path, file_name)
            # [한글 경로 우회] cv2.imread()는 윈도우에서 경로에 한글이 섞이면
            # 조용히 실패하는 버그가 있음 (예: "화이트_1" 폴더).
            # np.fromfile로 바이트를 직접 읽고 cv2.imdecode로 디코딩하면 한글 경로도 정상 작동함
            try:
                file_bytes = np.fromfile(file_path, dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                image = None
            if image is None:
                skipped += 1
                continue

            patch = extract_digit_patch(image)
            if patch is None:
                skipped += 1
                continue

            patches.append(patch)
            labels.append(digit_label)

    print(f"총 {len(patches)}장 로드 완료, {skipped}장은 숫자 못 찾아서 제외됨")
    return patches, labels


# =============================================================
# [Dataset 클래스] 파이토치가 학습 중에 데이터를 하나씩 꺼내가는 통로
# =============================================================
class DigitDataset(Dataset):
    def __init__(self, patches, labels, augment=False):
        self.patches = patches
        self.labels = labels
        self.augment = augment

        # [augmentation] 학습용 데이터에만 적용 - 매 epoch마다 랜덤하게 다르게 변형돼서
        # 데이터가 적어도 CNN이 다양한 각도/위치에 강해지도록 만들어줌
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(180),  # -180~180도 랜덤 회전 (뚜껑이 어느 각도로 돌아가 있어도 대응)
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),  # 살짝 이동/확대축소
                transforms.ToTensor(),  # 0~255 값을 0~1 사이 float 텐서로 변환
            ])
        else:
            # 검증용 데이터는 변형 없이 원본 그대로 사용 (진짜 실력을 정확히 측정하려고)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image = self.patches[idx]
        label = self.labels[idx]
        image_tensor = self.transform(image)
        return image_tensor, label


# =============================================================
# [CNN 모델] 간단한 3단 합성곱 구조
# =============================================================
class DigitCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # [features] 이미지에서 특징을 뽑아내는 부분
        # Conv2d: 필터로 무늬(모서리, 곡선 등)를 훑으면서 특징 지도를 만듦
        # MaxPool2d: 이미지 크기를 절반으로 줄이면서 중요한 특징만 남김
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        # [classifier] 뽑아낸 특징을 보고 최종적으로 1/2/3 중 뭔지 판단하는 부분
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.3),  # 학습 중 일부러 일부 연결을 꺼서 과적합(외우기) 방지
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================
# [메인 학습 루프]
# =============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 1) 데이터 로드 + 전처리
    print("데이터 로드 및 전처리 중...")
    patches, labels_str = load_dataset(TRAINING_DATA_DIR)

    if len(patches) == 0:
        print("[에러] 학습할 데이터가 하나도 없어. TRAINING_DATA_DIR 경로 확인해줘.")
        return

    # [라벨 매핑] "1","2","3" 문자열 라벨을 0,1,2 숫자 인덱스로 변환
    # (CNN 출력은 숫자 인덱스로 나오니까, 나중에 다시 "1","2","3"으로 되돌릴 표가 필요함)
    unique_labels = sorted(set(labels_str))  # ["1","2","3"]
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}
    labels_idx = [label_to_index[l] for l in labels_str]

    print(f"클래스: {unique_labels}")
    for label in unique_labels:
        count = labels_str.count(label)
        print(f"  숫자 {label}: {count}장")

    # 2) 학습용 / 검증용 나누기 (stratify로 클래스 비율 유지하면서 나눔)
    train_patches, val_patches, train_labels, val_labels = train_test_split(
        patches, labels_idx, test_size=VAL_RATIO, random_state=RANDOM_SEED, stratify=labels_idx
    )
    print(f"학습용 {len(train_patches)}장 / 검증용 {len(val_patches)}장")

    train_dataset = DigitDataset(train_patches, train_labels, augment=True)
    val_dataset = DigitDataset(val_patches, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3) 모델/손실함수/옵티마이저 준비
    model = DigitCNN(num_classes=len(unique_labels)).to(device)
    criterion = nn.CrossEntropyLoss()  # 여러 클래스 중 하나 맞히는 분류 문제에 쓰는 표준 손실함수
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    # 4) 학습 루프
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        # [검증] 학습에 안 쓴 데이터로 진짜 실력 확인
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
        val_acc = val_correct / len(val_dataset)

        print(f"[Epoch {epoch:2d}/{NUM_EPOCHS}] "
              f"학습 loss: {train_loss:.4f}, 학습 정확도: {train_acc*100:.1f}%, "
              f"검증 정확도: {val_acc*100:.1f}%")

        # [베스트 모델 저장] 검증 정확도가 지금까지 중 제일 높으면 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "digit_cnn_best.pth")

    print(f"\n학습 완료! 최고 검증 정확도: {best_val_acc*100:.1f}%")

    # 5) 라벨 매핑 정보 저장 (C#에서 결과 해석할 때 필요함)
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, ensure_ascii=False, indent=2)
    print("label_map.json 저장 완료")

    # 6) 제일 성능 좋았던 모델을 ONNX로 변환 (C#에서 쓸 최종 파일)
    model.load_state_dict(torch.load("digit_cnn_best.pth"))
    model.eval()
    dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    torch.onnx.export(
        model, dummy_input, "digit_cnn.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    print("digit_cnn.onnx 저장 완료 - 이제 C#에서 ONNX Runtime으로 불러다 쓰면 됨")


if __name__ == "__main__":
    main()
