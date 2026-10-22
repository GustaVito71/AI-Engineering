# Backend de negocio SIMULADO (Paso 12): imagen aparte y mínima, separada de la del
# servicio de IA. Representa un sistema externo cualquiera.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py data.json ./

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
