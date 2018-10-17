FROM alpine:3.8
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN apk add \
  bash \
  gcc \
  libffi-dev \
  musl-dev \
  openssl-dev \
  python3 \
  python3-dev \
  && pip3 install --no-cache-dir \
  flask \
  Flask_Sockets \
  kubernetes \
  'urllib3<1.24'

COPY notebook /notebook
COPY notebook-worker-image /notebook

EXPOSE 5000

WORKDIR /notebook
ENTRYPOINT ["python3", "notebook.py"]
