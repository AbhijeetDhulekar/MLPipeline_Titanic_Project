FROM python:3.11-slim-buster

WORKDIR /app


COPY requirements.txt .

RUN pip install -r requirements.txt


COPY src/ .
COPY models/ /app/models/
COPY models/titanic_model_latest.joblib /app/models/


EXPOSE 8000


CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]