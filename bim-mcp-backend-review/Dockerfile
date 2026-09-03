FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PLANT2BIM_YOLO_PYTHON=/usr/local/bin/python \
    PLANT2BIM_YOLO_WEIGHTS=/app/models/plan2bim/openings.pt \
    PLANT2BIM_YOLO_WALL_WEIGHTS=/app/models/plan2bim/walls.pt \
    PLANT2BIM_YOLO_WALL_SOURCE=hybrid \
    PLANT2BIM_YOLO_DEVICE=cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libx11-6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-plan2bim.txt ./
RUN pip install --no-cache-dir -r requirements-plan2bim.txt

COPY plan_to_bim_free_app.py ./
COPY plantatobim ./plantatobim
COPY bim_editing ./bim_editing
COPY experiments/plant2bim/run_pre_wall_opening_pipeline.py ./experiments/plant2bim/run_pre_wall_opening_pipeline.py
COPY models/plan2bim ./models/plan2bim

EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 300 --graceful-timeout 30 --access-logfile - --error-logfile - plan_to_bim_free_app:app"]
