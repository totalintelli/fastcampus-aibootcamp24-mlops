# fastcampus-aibootcamp24-mlops

패스트캠퍼스 AI Bootcamp 24 과정에서 진행한 MLOps 프로젝트입니다. 프로젝트 기간은 **2026년 2월 27일 ~ 2026년 3월 13일**이며, Python 중심의 머신러닝 추천 시스템, FastAPI 기반 API 서버, Dockerfile, 부하 테스트 스크립트, 모델 아티팩트 저장 구조가 포함되어 있습니다. ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops))

## 프로젝트 개요

이 프로젝트는 시청 로그 기반 데이터를 사용해 콘텐츠를 추천하는 간단한 머신러닝 파이프라인을 구현합니다. 코드 구조상 다음 흐름을 갖습니다. 학습 진입점(`src/main.py`)에서 데이터셋 로딩, 학습/검증/테스트, 모델 저장을 수행하고, 추론 결과는 DB에 기록하며, 웹 애플리케이션(`src/webapp.py`)은 단건 예측 API와 배치 추천 조회 API를 제공합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/main.py))

## 주요 기능

- **학습 파이프라인**
  - 데이터셋 분리(train/val/test)
  - 간단한 DataLoader 기반 배치 학습
  - 검증 및 테스트 수행
  - 모델 체크포인트 저장 ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/main.py))

- **추천 추론**
  - 저장된 체크포인트 로드
  - 사용자 입력 기반 추천 생성
  - 추천 결과를 DB에 저장 ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops/blob/main/src/inference/inference.py))

- **API 서비스**
  - `POST /predict` : 입력 데이터 기반 추천
  - `GET /batch-predict?k=5` : DB에 저장된 최근 추천 결과 조회 ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/webapp.py))

- **실험 및 운영 보조**
  - Weights & Biases 연동
  - Docker 기반 실행 준비
  - `wrk`를 이용한 부하 테스트 스크립트 제공
  - 모델 파일 SHA-256 해시 저장/검증 로직 포함 ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops/blob/main/requirements.txt))

## 프로젝트 구조

저장소에는 아래와 같은 구조가 포함되어 있습니다. 루트에는 `models/movie_predictor`, `opt`, `src`, `Dockerfile`, `requirements.txt`, `start_api_server.sh`, `stress-test.sh` 등이 있고, `src` 아래에는 `dataset`, `evaluate`, `inference`, `model`, `postprocess`, `train`, `utils`, `main.py`, `webapp.py`가 있습니다. `models/movie_predictor` 경로에는 여러 `.pkl` 모델 파일과 일부 `.sha256` 검증 파일이 존재합니다. ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops))

```text
fastcampus-aibootcamp24-mlops/
├── models/
│   └── movie_predictor/
├── opt/
├── src/
│   ├── dataset/
│   ├── evaluate/
│   ├── inference/
│   ├── model/
│   ├── postprocess/
│   ├── train/
│   ├── utils/
│   ├── .env
│   ├── main.py
│   └── webapp.py
├── Dockerfile
├── install_wrk.sh
├── requirements.txt
├── start_api_server.sh
└── stress-test.sh
```

## 기술 스택

- **Language**: Python
- **API**: FastAPI, Uvicorn
- **ML/Data**: NumPy, Pandas, scikit-learn
- **Experiment Tracking**: Weights & Biases
- **DB Access**: SQLAlchemy, mysqlclient
- **Container**: Docker
- **Load Test**: wrk ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops))

## 동작 방식

### 1) 데이터 처리

`watch_log.py`는 시청 로그 CSV를 읽고, `rating`, `popularity`, `watch_seconds`를 feature로 사용하며 `content_id`를 label로 인코딩합니다. 또한 `train_test_split`을 사용해 학습/검증/테스트 데이터셋으로 나눕니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/dataset/watch_log.py))

### 2) 모델 학습

`main.py`의 `train` 명령은 데이터셋을 불러오고, `ModelFactory`를 통해 모델을 생성한 뒤, 학습과 검증을 반복합니다. 학습 로그는 Weights & Biases에 기록되며, 마지막에는 테스트 평가 후 모델이 저장됩니다. 현재 팩토리에는 `movie_predictor` 모델이 등록되어 있습니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/main.py))

### 3) 모델 저장 및 검증

모델은 `models/<model_name>/` 경로에 `E{epoch}_T{timestamp}.pkl` 형식으로 저장되며, 별도 SHA-256 해시 파일을 생성해 무결성을 확인할 수 있게 되어 있습니다. 저장소에도 실제 `.pkl` 및 `.sha256` 파일이 포함되어 있습니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/model/movie_predictor.py))

### 4) 추론 및 서비스

`webapp.py`는 FastAPI 앱을 실행하며, `/predict`는 사용자 입력을 받아 추천 결과를 반환하고 `/batch-predict`는 DB에서 최근 추천 결과를 읽어 반환합니다. `start_api_server.sh`는 `python src/webapp.py`를 백그라운드로 실행합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/webapp.py))

## 설치 방법

### 1) 저장소 클론

```bash
git clone https://github.com/totalintelli/fastcampus-aibootcamp24-mlops.git
cd fastcampus-aibootcamp24-mlops
```

### 2) 가상환경 생성 및 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

의존성 파일에는 FastAPI, Uvicorn, NumPy, Pandas, scikit-learn, SQLAlchemy, mysqlclient, wandb 등이 포함되어 있습니다. ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops/blob/main/requirements.txt))

## 환경 변수 설정

저장소에는 `src/.env` 파일이 존재하며, 코드상 다음 환경 변수를 사용합니다.

- `TMDB_BASE_URL`
- `TMDB_API_KEY`
- `WANDB_API_KEY`
- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_PORT` ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/.env))

예시:

```bash
# src/.env 예시
TMDB_BASE_URL=YOUR_TMDB_BASE_URL
TMDB_API_KEY=YOUR_TMDB_API_KEY
WANDB_API_KEY=YOUR_WANDB_API_KEY
DB_USER=YOUR_DB_USER
DB_PASSWORD=YOUR_DB_PASSWORD
DB_HOST=YOUR_DB_HOST
DB_PORT=3306
```

> 보안상 실제 API 키나 비밀번호는 절대 저장소에 커밋하지 않는 것을 권장합니다.

## 학습 실행

`src/main.py`는 Fire 기반 CLI 엔트리포인트로 `preprocessing`, `train`, `inference` 명령을 제공합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/main.py))

### 전처리 실행

```bash
python src/main.py preprocessing --date=260312
```

### 학습 실행

```bash
python src/main.py train --model_name=movie_predictor --num_epochs=10 --lr=0.01
```

설명:
- `model_name`: 현재는 `movie_predictor` 사용
- `num_epochs`: 학습 epoch 수
- `lr`: 학습률 파라미터로 전달됨 ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/main.py))

## API 서버 실행

### 직접 실행

```bash
python src/webapp.py
```

기본적으로 `0.0.0.0:8000`에서 실행됩니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/webapp.py))

### 스크립트 실행

```bash
bash start_api_server.sh
```

([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops/blob/main/start_api_server.sh))

## API 사용 예시

### 1) 단건 추천

**POST** `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "content_id": 101,
    "watch_seconds": 1200,
    "rating": 4.5,
    "popularity": 88.2
  }'
```

예상 응답 형식:

```json
{
  "recommended_content_id": [123, 456, 789]
}
```

입력 스키마는 `user_id`, `content_id`, `watch_seconds`, `rating`, `popularity`로 정의되어 있습니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/webapp.py))

### 2) 배치 추천 조회

**GET** `/batch-predict?k=5`

```bash
curl "http://localhost:8000/batch-predict?k=5"
```

예상 응답 형식:

```json
{
  "recommended_content_id": [123, 456, 789, 321, 654]
}
```

`stress-test.sh` 역시 이 엔드포인트를 대상으로 요청하도록 작성되어 있습니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/webapp.py))

## Docker 실행

저장소의 Dockerfile은 `python:3.11-bookworm` 이미지를 기반으로 의존성을 설치하고, `dataset`, `src`, `start_api_server.sh`를 `/opt` 아래로 복사하도록 작성되어 있습니다. 작업 디렉터리는 `/opt`입니다. ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops/blob/main/Dockerfile))

예시 빌드:

```bash
docker build -t fastcampus-mlops .
```

예시 실행:

```bash
docker run -p 8000:8000 --env-file src/.env fastcampus-mlops python src/webapp.py
```

> 참고: Dockerfile 주석에도 `.env` 파일이 컨테이너에 직접 복사되지 않는다는 점이 드러나므로, 실행 시 `--env-file` 또는 환경 변수 주입 방식으로 전달하는 편이 적절합니다. ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops/blob/main/Dockerfile))

## 부하 테스트

`install_wrk.sh`는 `wrk`를 설치하고, `stress-test.sh`는 30초 동안 `http://localhost:8000/batch-predict?k=5` 엔드포인트에 대해 2개 스레드, 2개 커넥션으로 지연시간 포함 테스트를 수행합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/install_wrk.sh))

```bash
bash install_wrk.sh
bash stress-test.sh
```

## 모델 구현 메모

현재 `movie_predictor`는 NumPy 기반의 간단한 2-layer 신경망 형태로 구현되어 있으며, `relu`, `softmax`, `forward`, `backward` 메서드를 포함합니다. 학습 손실과 평가 손실은 코드상 평균제곱오차 형태로 계산됩니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/model/movie_predictor.py))

## 주의 사항

- DB 읽기/쓰기 기능은 MySQL 연결 정보를 필요로 하므로 DB가 준비되어 있어야 합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/postprocess/postprocess.py))
- 시청 로그 데이터는 `watch_log.csv`를 기준으로 읽도록 구현되어 있습니다. 경로는 `opt/mlops/dataset/watch_log.csv`로 지정되어 있으므로 실제 실행 전 데이터 파일 위치를 확인해야 합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/dataset/watch_log.py))
- 저장소 내 `.env` 파일에 민감 정보가 포함될 수 있으므로 실제 운영에서는 비밀정보 관리 도구 사용을 권장합니다. ([raw.githubusercontent.com](https://raw.githubusercontent.com/totalintelli/fastcampus-aibootcamp24-mlops/refs/heads/main/src/.env))

## 라이선스

이 저장소는 **MIT License**를 사용합니다. ([github.com](https://github.com/totalintelli/fastcampus-aibootcamp24-mlops))

---
