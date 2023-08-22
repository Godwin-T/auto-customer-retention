FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /home

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY app.py .


EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "app:app" ]

#WORKDIR /home/godwin/Documents/Workflow/Churn-Prediction-in-a-Telecom-Company

#COPY "requirements.txt" /app/

#RUN pip install -r /app/requirements.txt