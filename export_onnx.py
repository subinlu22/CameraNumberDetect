# -*- coding: utf-8 -*-
"""
이미 학습 끝난 digit_cnn_best.pth를 다시 학습하지 않고
ONNX 파일로만 변환하는 스크립트.

(train_digit_cnn.py 돌릴 때 onnx 패키지가 없어서 마지막 변환 단계만
 실패했던 것 - 학습 결과(pth)는 이미 저장되어 있으니 재학습 필요 없음)

■ 실행 전 준비
    C:\\Users\\dev\\venv311\\Scripts\\python.exe -m pip install onnx --break-system-packages

■ 실행 방법
    C:\\Users\\dev\\venv311\\Scripts\\python.exe export_onnx.py
"""

import json
import torch
import torch.nn as nn

IMAGE_SIZE = 64


# train_digit_cnn.py에 있던 것과 완전히 똑같은 모델 구조
# (구조가 다르면 저장된 가중치를 못 불러오니까 반드시 일치해야 함)
class DigitCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    with open("label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    print(f"클래스 개수: {num_classes}, 매핑: {label_map}")

    model = DigitCNN(num_classes=num_classes)
    # weights_only=True: 신뢰 안 되는 pickle 실행 경고 없애는 안전한 로드 방식
    model.load_state_dict(torch.load("digit_cnn_best.pth", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, dummy_input, "digit_cnn.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    print("digit_cnn.onnx 저장 완료!")


if __name__ == "__main__":
    main()
