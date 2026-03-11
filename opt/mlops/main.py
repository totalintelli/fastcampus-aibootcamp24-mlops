'''
파일 명칭 : main.py
기     능 : 신경망 모델로 콘텐츠를 추천한다.
입     력 : 평점(rating), 인기도(popularity), 시청 시간(watch_seconds)
출     력 : 추천 콘텐츠(content_id)
작 성  자 : 송 용 단
작성 일자 : 2026-03-11
'''
import numpy as np
import pandas as pd

# 데이터 로드
df = pd.read_csv('./dataset/watch_log.csv')
columns = ["rating", "popularity", "watch_seconds", "content_id"]
df = df[columns].drop_duplicates()

# content_id 인코딩 (클래스화)
content_ids = df["content_id"].astype("category")
df["content_label"] = content_ids.cat.codes   # 0,1,2,3,...
num_classes = df["content_label"].nunique()

x = df[["rating", "popularity", "watch_seconds"]].values
y = df["content_label"].values.reshape(-1, 1)

# 데이터 분할
idx = np.random.permutation(len(x))
x = x[idx]
y = y[idx]

split = int(len(x) * 0.8)
x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]

# 유틸리티 함수
def one_hot(y, num_classes):
    o_h = np.zeros((y.shape[0], num_classes))
    o_h[np.arange(y.shape[0]), y.flatten()] = 1
    return o_h

y_train_oh = one_hot(y_train, num_classes)
y_val_oh = one_hot(y_val, num_classes)

# 신경망 모델 정의
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = x @ self.w1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.out = self.softmax(self.z2)
        return self.out

    def backward(self, x, y_true, lr=0.001):
        m = y_true.shape[0]

        # softmax + cross entropy gradient
        dz2 = (self.out - y_true) / m
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

# 모델 초기화
model = SimpleNN(input_dim=3, hidden_dim=128, output_dim=num_classes)

# 학습 루프
epochs = 15
lr = 0.001

for epoch in range(epochs):
    pred = model.forward(x_train)

    # Cross Entropy Loss
    train_loss = -np.mean(np.sum(y_train_oh * np.log(pred + 1e-9), axis=1))

    model.backward(x_train, y_train_oh, lr)

    val_pred = model.forward(x_val)
    val_loss = -np.mean(np.sum(y_val_oh * np.log(val_pred + 1e-9), axis=1))

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
