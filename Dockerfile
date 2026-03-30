# Full repo image: tracker OR worker (same dependencies).
#
# Build:
#   docker build -t gpu_p2p .
#
# Run TRACKER (control plane):
#   docker run --rm -p 8000:8000 \
#     -e PYTHONPATH=/app \
#     gpu_p2p uvicorn tracker.app:app --host 0.0.0.0 --port 8000
#
# Optional: mount Fashion-MNIST CSV dir for tracker global eval (same as local FASHION_MNIST_CSV_DIR):
#   docker run --rm -p 8000:8000 \
#     -v "/absolute/path/to/archive 2:/data/fashion:ro" \
#     -e PYTHONPATH=/app \
#     -e FASHION_MNIST_CSV_DIR=/data/fashion \
#     gpu_p2p uvicorn tracker.app:app --host 0.0.0.0 --port 8000
#
# Run WORKER (training container default CMD):
#   docker run --rm ... (set TRACKER_URL, WORKER_ID, TASK_JSON per trainer_wrapper)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY worker /app/worker
COPY tracker /app/tracker
COPY mock_worker.py /app/mock_worker.py

ENV PYTHONPATH=/app
CMD ["python", "-u", "-m", "worker.trainer_wrapper"]
