'''
파일명칭 : data_loader.py
기    능 : 머신러닝 학습할 때 필요한 데이터 셋을 다양한 형태로 다룬다.
입    력 : 데이터를 다루기 위한 기준들
출    력 : 기준에 해당하는 데이터
작성  자 : 송 용 단
작성일자 : 2026-03-11
'''
import math

import numpy as np


class SimpleDataLoader:
    def __init__(self, features, labels, batch_size=32, shuffle=True):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(features)
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        self.current_idx = end_idx

        batch_indices = self.indices[start_idx:end_idx]
        return self.features[batch_indices], self.labels[batch_indices]

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)
