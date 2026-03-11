'''
파일명칭 : evaluate.py 
기    능 : 모델 추론 및 평가 루프
입    력 : 모델, value loader
출    력 : 총 손실률, 모든 예측값
작성  자 : 송 용 단
작성일자 : 2026-03-11
'''
import numpy as np


def evaluate(model, val_loader):
    total_loss = 0
    all_predictions = []

    for features, labels in val_loader:
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)

        loss = np.mean((predictions - labels) ** 2)
        total_loss += loss * len(features)

        predicted = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted)

    return total_loss / len(val_loader), all_predictions
