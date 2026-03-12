"""
파일명칭 : webapp.py
기    능 : 영화 추천 목록 화면을 웹 사이트에 표시한다.
입    력 : 추천 콘텐츠 아이디
출    력 : 영화 추천 목록 화면
작 성 자 : 송 용 단
작성일자 : 2026-03-12
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference.inference import inference, init_model, load_checkpoint
from src.postprocess.postprocess import read_db

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

load_dotenv()
checkpoint = load_checkpoint()
model, scaler, label_encoder = init_model(checkpoint)


class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float


@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = np.array(
            [
                input_data.user_id,
                input_data.content_id,
                input_data.watch_seconds,
                input_data.rating,
                input_data.popularity,
            ]
        )
        recommend = inference(model, scaler, label_encoder, data)
        recommend = [int(r) for r in recommend]
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch-predict")
async def batch_predict(k: int = 5):
    try:
        recommend = read_db("mlops", "recommend", k=k)
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
