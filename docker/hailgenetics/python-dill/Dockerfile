ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}
RUN pip install --upgrade --no-cache-dir dill numpy scipy scikit-learn && \
    python3 -m pip check && \
    apt-get update && \
    apt-get install -y \
      libopenblas0 \
    && rm -rf /var/lib/apt/lists/*
