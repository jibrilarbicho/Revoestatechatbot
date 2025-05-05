# Use a slim Python 3.10 base image
FROM python:3.10-slim
WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip show gunicorn && which gunicorn || echo "ERROR: gunicorn not found"
EXPOSE 7860
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:7860"]