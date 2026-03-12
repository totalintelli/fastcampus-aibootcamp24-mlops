"""
파일명칭 : postprocess.py
기    능 : 데이터 저장(후처리) 클라이언트 코드
입    력 : 데이터, 데이터메이스 이름, 표 이름
출    력 :
작 성 자 : 송 용 단
작성일자 : 2026-03-12
"""

import os

import pandas as pd
from sqlalchemy import create_engine


def write_db(data: pd.DataFrame, db_name, table_name):
    engine = create_engine(
        url=(
            f"mysql+mysqldb://"
            f"{os.environ.get('DB_USER')}:"
            f"{os.environ.get('DB_PASSWORD')}@"
            f"{os.environ.get('DB_HOST')}:"
            f"{os.environ.get('DB_PORT')}/"
            f"{db_name}"
        )
    )

    connect = engine.connect()
    data.to_sql(table_name, connect, if_exists="append")
    connect.close()
