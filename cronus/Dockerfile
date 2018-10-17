FROM alpine:3.8
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN apk add \
  bash \
  python3 \
  && pip3 install --no-cache-dir \
  flask \
  kubernetes

COPY cronus /cronus

EXPOSE 5000

CMD ['python', '/cronus/cronus.py']
