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
  # >=19.0.2 due to https://github.com/pypa/pip/issues/6197#issuecomment-462014853
  pip3 install -U 'pip>=19.0.2' && \
  pip3 install --no-cache-dir \
  flask \
  Flask_Sockets \
  kubernetes \
  'urllib3<1.24'

COPY notebook /notebook
COPY notebook-worker-images /notebook

EXPOSE 5000

WORKDIR /notebook
ENTRYPOINT ["python3", "notebook.py"]
