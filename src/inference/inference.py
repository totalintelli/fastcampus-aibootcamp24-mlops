"""
파일명칭 : inference.py
기    능 : 학습된 모델로 사용자 선호 콘텐츠를 추론
입    력 : 학습한 모델
출    력 : 사용자 선호 콘텐츠 아이디
작 성 자 : 송 용 단
작성일자 : 2026-03-12
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.model.movie_predictor import MoviePredictor
from src.utils.utils import calculate_hash, model_dir, read_hash


def make_inference_df(data):
    columns = "user_id content_id watch_seconds rating popularity".split()
    return pd.DataFrame(data=[data], columns=columns)


def model_validation(model_path):
    original_hash = read_hash(model_path)
    current_hash = calculate_hash(model_path)
    if original_hash == current_hash:
        print("validation success")
        return True
    else:
        return False


def load_checkpoint():
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
    latest_model = glob.glob(models_path)[-1]

    if model_validation(latest_model):
        with open(latest_model, "rb") as f:
            checkpoint = pickle.load(f)

        return checkpoint
    else:
        raise FileExistError("Not found or invalid model file")


def init_model(checkpoint):
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoder = checkpoint.get("label_encoder", None)
    return model, scaler, label_encoder
