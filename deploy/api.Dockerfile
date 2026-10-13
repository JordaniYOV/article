FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PLANETARY_WORKSPACE_ROOT=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements/ requirements/
RUN python -m pip install --no-cache-dir -r requirements/api.txt
COPY pyproject.toml ./
COPY src/api/ src/api/
COPY src/planetary_vlm/ src/planetary_vlm/
COPY src/benchmark_services/__init__.py src/benchmark_services/__init__.py
COPY src/benchmark_services/common/ src/benchmark_services/common/
# Existing bounded orbital converters and pinned recovery metadata, not datasets.
COPY configs/ configs/
COPY scripts/*.py scripts/
RUN python -m pip install --no-cache-dir --no-deps -e .
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
