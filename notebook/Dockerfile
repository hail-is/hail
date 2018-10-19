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
  kubernetes \
  'urllib3<1.24'

COPY notebook /notebook

EXPOSE 5000

ENTRYPOINT ["python3", "/notebook/notebook.py"]
