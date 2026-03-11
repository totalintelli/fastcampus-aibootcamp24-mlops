'''
파일 명칭 : utils.py 
기     능 : 유틸리티성 코드 모듈화
입     력 : 없음
출     력 : 없음
작 성  자 : 송 용 단
작성 일자 : 2026-03-11
'''
import os
import random

import numpy as np


def init_seed():
    np.random.seed(0)
    random.seed(0)


def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )


def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )

# Run name 자동 지정하기
def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)
