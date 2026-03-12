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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.model.movie_predictor import MoviePredictor
from src.utils.utils import model_dir


def load_checkpoint():
    target_dir = model_dir(MoviePredictor.name)
    models_path = os.path.join(target_dir, "*.pkl")
    latest_model = glob.glob(models_path)[-1]

    with open(latest_model, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint


def init_model(checkpoint):
    model = MoviePredictor(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = checkpoint.get("scaler", None)
    label_encoder = checkpoint.get("label_encoder", None)
    return model, scaler, label_encoder
