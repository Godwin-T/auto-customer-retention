FROM python:3.9.7-slim

RUN pip install -U pip

WORKDIR /home

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["app.py", "utils.py" , "./"]

EXPOSE 9696

#VOLUME /home/mlruns

VOLUME /home/mlflow.db

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]
