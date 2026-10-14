FROM python:3.12-slim
ARG SERVICE
ARG PROFILE=mock
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PLANETARY_WORKSPACE_ROOT=/workspace
ENV MODEL_SERVICE=${SERVICE}
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements/ requirements/
# No model weights in images. GPU dependency builds are explicit and NOT tested.
RUN case "$SERVICE" in spacellava|ibm) ;; *) exit 2 ;; esac; \
    if [ "$PROFILE" = "mock" ]; then \
      python -m pip install --no-cache-dir -r requirements/api.txt; \
    elif [ "$PROFILE" = "gpu" ]; then \
      if [ "$SERVICE" = "spacellava" ]; then \
        python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 && \
        python -m pip install --no-cache-dir --only-binary=tokenizers -r requirements/spacellava.txt; \
      else \
        python -m pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130 && \
        python -m pip install --no-cache-dir -r requirements/ibm-imp.txt; \
      fi; \
    else exit 2; fi
COPY pyproject.toml ./
COPY src/planetary_vlm/ src/planetary_vlm/
COPY src/benchmark_services/__init__.py src/benchmark_services/__init__.py
COPY src/benchmark_services/common/ src/benchmark_services/common/
# Workers reuse only the lightweight logging module, not public API routes.
COPY src/api/__init__.py src/api/logging.py src/api/
# Each image contains only its own model package, never the other model or API routes.
COPY src/benchmark_services/${SERVICE}/ src/benchmark_services/${SERVICE}/
RUN python -m pip install --no-cache-dir --no-deps -e . && \
    git config --system --add safe.directory /workspace/external/LLaVA && \
    git config --system --add safe.directory /workspace/external/NASA-IBM-Lunar-Foundation-Model
ENTRYPOINT ["python", "-m", "benchmark_services.common.server"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
