'''
파일명칭 : train.py 
기    능 : 모델 학습 루프 구현
입    력 : 머신러닝 모델, train loader
출    력 : 총 손실률
작성  자 : 송 용 단
작성일자 : 2026-03-11
'''
import numpy as np


def train(model, train_loader):
    total_loss = 0
    for features, labels in train_loader:
        predictions = model.forward(features)
        labels = labels.reshape(-1, 1)
        loss = np.mean((predictions - labels) ** 2)

        model.backward(features, labels, predictions)

        total_loss += loss

    return total_loss / len(train_loader)
