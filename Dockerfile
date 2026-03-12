FROM python:3.11-bookworm

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY dataset /opt/dataset
COPY src /opt/src
COPY start_api_server.sh /opt
# .env 파일은 왜 컨테이너 COPY 구문에 포함되지 않을까요?

WORKDIR /opt
