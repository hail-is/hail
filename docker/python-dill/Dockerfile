FROM python:@PYTHON_VERSION@
RUN pip install --use-feature=2020-resolver --upgrade --no-cache-dir dill numpy scipy sklearn && \
    python3 -m pip check && \
    apt-get update && \
    apt-get install -y \
      libopenblas-base \
    && rm -rf /var/lib/apt/lists/*
