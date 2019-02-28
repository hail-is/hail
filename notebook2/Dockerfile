FROM alpine:3.8
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN apk add \
  bash \
  gcc \
  libffi-dev \
  musl-dev \
  openssl-dev \
  python3 \
  python3-dev && \
  # <19 due to https://github.com/pypa/pip/issues/6197
  pip3 install -U 'pip<19' && \
  pip3 install --no-cache-dir \
  flask \
  Flask_Sockets \
  kubernetes \
  'urllib3<1.24'\
  libsass \
  authlib

COPY notebook /notebook
COPY notebook-worker-images /notebook

EXPOSE 5000

WORKDIR /notebook
ENTRYPOINT ["python3", "notebook.py"]
