# Use Python 3.10 slim base image
FROM python:3.10-slim
WORKDIR /app
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache
ENV HF_HOME=/app/.cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && chmod -R 777 /app/.cache
COPY GeminiAgent.py main.py routes.py serialization.py tool.py .
EXPOSE 7860
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:7860"]