FROM python:3.11-slim

WORKDIR /app

# Menginstal dependensi sistem yang mungkin dibutuhkan oleh library AI/Vector DB
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pastikan path python mengenali folder src
ENV PYTHONPATH=/app

# Eksekusi FastAPI di port 7860
CMD ["uvicorn", "src.production_rag.api.main:app", "--host", "0.0.0.0", "--port", "7860"]