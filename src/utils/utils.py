"""
파일 명칭 : utils.py
기     능 : 유틸리티성 코드 모듈화
입     력 : 없음
출     력 : 없음
작 성  자 : 송 용 단
작성 일자 : 2026-03-11
"""

import hashlib
import os
import random
from datetime import datetime

import numpy as np

"""
진입점(Entrypoint) 매개변수 조정
- 데이터 전처리 등의 태스크에서는 특정 날짜에 의존적인 경우가 대부분임.
  - 특히 배치 학습, 배치 추론의 경우
- 따라서 전처리 태스크에 날짜 정보를 수행 파라미터로 추가
"""


def parse_date(date: str):
    date_format = "%y%m%d"
    parsed_date = datetime.strptime(str(date).replace("-", ""), date_format)
    return parsed_date


"""
## 모델 파일 검증 추가하기

- `torch.save` 는 내부적으로 pickle을 사용해 직렬화(마샬링) 후 저장하게 됨
    - 이는 보안적 취약점으로 작용하게 됨. 따라서 최소한의 검증 절차가 필요
    - 아래 실습에서는 sha256 해시 알고리즘을 통해 변조 여부를 최소한으로 확인
"""


def calculate_hash(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_hash(dst):
    hash_ = calculate_hash(dst)
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "w") as f:
        f.write(hash_)


def read_hash(dst):
    hash_file = f"{dst}.sha256"

    # 해시 파일이 없으면 생성
    if not os.path.exists(hash_file):
        print(f"Hash file not found. Creating: {hash_file}")
        import hashlib

        with open(dst, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        with open(hash_file, "w") as f:
            f.write(file_hash)
        return file_hash

    with open(hash_file, "r") as f:
        return f.read().strip()


def init_seed():
    np.random.seed(0)
    random.seed(0)


def project_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def model_dir(model_name):
    return os.path.join(project_path(), "models", model_name)


# Run name 자동 지정하기
def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)
