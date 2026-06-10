FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8010

# 1 worker para TensorFlow/Keras: evita duplicar la carga del modelo en memoria
# Con 16 GiB en t3a.xlarge, 1 worker es suficiente y mas estable
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
