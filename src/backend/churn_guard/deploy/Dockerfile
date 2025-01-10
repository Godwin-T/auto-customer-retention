FROM python:3.9.7-slim

RUN pip install -U pip

WORKDIR /home

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip/http \
    python3 -m pip install -U pip==${PIP_VERSION}

COPY app.py .

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]
