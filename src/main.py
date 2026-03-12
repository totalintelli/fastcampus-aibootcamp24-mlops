"""
파일명칭 : main.py
기    능 : 메인 진입점. 전체과정(데이터셋 준비 -> 모델 준비 -> 학습 -> 검증 및 평가 -> (모델 저장)) 일괄 수행
입    력 :
출    력 :
작 성 자 : 송 용 단
작성일자 : 2026-03-11
"""

import os
import sys

import fire
import numpy as np
from dotenv import load_dotenv

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.data_loader import SimpleDataLoader
from src.dataset.watch_log import get_datasets
from src.evaluate.evaluate import evaluate
from src.inference.inference import (
    inference,
    init_model,
    load_checkpoint,
    recommend_to_df,
)
from src.model.movie_predictor import MoviePredictor, model_save
from src.postprocess.postprocess import write_db
from src.train.train import train
from src.utils.factory import ModelFactory
from src.utils.utils import auto_increment_run_suffix, init_seed, model_dir, parse_date

"""
# 진입점(Entrypoint) 매개변수 조정

- 데이터 전처리 등의 태스크에서는 특정 날짜에 의존적인 경우가 대부분임.
  (특히 배치 학습, 배치 추론의 경우) 
  따라서 전처리 태스크에 날짜 정보를 수행 파라미터로 추가
"""


def run_preprocessing(date):
    parsed_date = parse_date(date)
    print(f"Run date : {parsed_date.year}. {parsed_date.month}. {parsed_date.day}")
    print("Run some preprocessing...")
    print("Done!")


# 추론 태스크 추가
def run_inference(data=None, batch_size=64):
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)

    if data is None:
        data = []

    data = np.array(data)

    recommend = inference(model, scaler, label_encoder, data, batch_size)
    print(recommend)

    write_db(recommend_to_df(recommend), "mlops", "recommend")


init_seed()
load_dotenv()


def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")


def get_latest_run(project_name):
    runs = get_runs(project_name)
    if not runs:
        return f"{project_name}-000"

    return runs[0].name


def run_train(model_name, num_epochs=10, lr=0.01, model_ext="pth"):
    # 데이터셋 및 DataLoader 생성
    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(
        train_dataset.features, train_dataset.labels, batch_size=64, shuffle=True
    )
    val_loader = SimpleDataLoader(
        val_dataset.features, val_dataset.labels, batch_size=64, shuffle=False
    )
    test_loader = SimpleDataLoader(
        test_dataset.features, test_dataset.labels, batch_size=64, shuffle=False
    )

    # 모델 초기화
    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
        "hidden_dim": 64,
    }
    model = ModelFactory.create(model_name, **model_params)

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    project_name = model_name.replace("_", "-")

    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="content-based movie recommend model",
        tags=["content-based", "movie", "recommend"],
        config=locals(),
    )

    # 학습 루프
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader)
        val_loss, _ = evaluate(model, val_loader)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val-Train Loss : {val_loss-train_loss:.4f}"
        )
        wandb.log({"Loss/Train": train_loss})
        wandb.log({"Loss/valid": val_loss})

    # 테스트
    test_loss, predictions = evaluate(model, test_loader)
    print(f"Test Loss : {test_loss:.4f}")
    print([train_dataset.decode_content_id(idx) for idx in predictions])

    model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        loss=train_loss,
        scaler=train_dataset.scaler,
        label_encoder=train_dataset.label_encoder,
    )

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(
        {
            "preprocessing": run_preprocessing,
            "train": run_train,
            "inference": run_inference,
        }
    )
