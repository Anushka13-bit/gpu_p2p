# Worker training image (GPU). Build: docker build -t novacompute-worker .
# Run via worker.docker_manager.run_training_container with project mount to /app.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic python-multipart \
    gputil requests docker

COPY shared /app/shared
COPY worker /app/worker

ENV PYTHONPATH=/app
CMD ["python", "-u", "-m", "worker.trainer_wrapper"]
