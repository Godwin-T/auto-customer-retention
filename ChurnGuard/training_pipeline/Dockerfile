FROM python:3.9.7-slim

RUN pip install -U pip

WORKDIR /home

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /home/

RUN chmod +x /home/run.sh

ENTRYPOINT [ "/home/run.sh"]
