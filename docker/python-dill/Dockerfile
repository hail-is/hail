FROM python:@PYTHON_VERSION@
RUN pip install -U --no-cache-dir dill numpy scipy sklearn && \
    apt-get update && \
    apt-get install -y \
      libopenblas-base \
    && rm -rf /var/lib/apt/lists/*
