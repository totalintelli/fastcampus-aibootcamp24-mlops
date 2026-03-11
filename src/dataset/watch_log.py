'''
파일 명칭 : watch_log.py
기     능 : 데이터셋 관리(데이터셋 준비 -> 모델 준비 -> 학습 -> 검증 및 평가 -> 모델 저장)
입     력 : TMDB 원본 데이터
출     력 : TMDB 데이터셋
작 성   자 : 송 용 단
작성 일자 : 2026-03-11
'''
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.utils import project_path


class WatchLogDataset:
    def __init__(self, df, scaler=None, label_encoder=None):
        self.df = df
        self.features = None
        self.labels = None
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.contents_id_map = None
        self._preprocessing()

    def _preprocessing(self):
        # content_id를 정수형으로 변환
        if self.label_encoder:
            # [수정] 훈련 데이터에 없던 라벨(unseen labels)을 찾아내어 필터링하거나 처리
            # 여기서는 안전하게 '이미 학습된 라벨' 안에 있는 데이터만 남기는 방식을 사용합니다.
            known_labels = self.label_encoder.classes_

            # 학습되지 않은 라벨이 포함된 행을 제거 (또는 특정 값으로 치환)
            mask = self.df["content_id"].isin(known_labels)
            if not mask.all():
                unseen = self.df.loc[~mask, "content_id"].unique()
                print(f"[Warning] Removing rows with unseen labels: {unseen}")
                self.df = self.df[mask].copy() # unseen label 제거

            self.df["content_id"] = self.label_encoder.transform(self.df["content_id"])
        else:
            self.label_encoder = LabelEncoder()
            self.df["content_id"] = self.label_encoder.fit_transform(self.df["content_id"])

        # content_id 디코딩 맵 생성
        self.contents_id_map = dict(enumerate(self.label_encoder.classes_))

        # 타겟 및 피처 정의 (self.df가 필터링되었을 수 있으므로 다시 할당)
        target_columns = ["rating", "popularity", "watch_seconds"]
        self.labels = self.df["content_id"].values
        features = self.df[target_columns].values

        # 피처 스케일링
        if self.scaler:
            self.features = self.scaler.transform(features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)


    def decode_content_id(self, encoded_id):
        return self.contents_id_map[encoded_id]

    @property
    def features_dim(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_dataset():
    watch_log_path = os.path.join(project_path(), "dataset", "watch_log.csv")
    return pd.read_csv(watch_log_path)


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets(scaler=None, label_encoder=None):
    df = read_dataset()
    train_df, val_df, test_df = split_dataset(df)
    train_dataset = WatchLogDataset(train_df, scaler, label_encoder)
    val_dataset = WatchLogDataset(val_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    test_dataset = WatchLogDataset(test_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    return train_dataset, val_dataset, test_dataset
