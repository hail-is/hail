FROM alpine:3.8
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN apk add \
  bash \
  curl \
  gcc \
  htop \
  libffi-dev \
  make \
  musl-dev \
  openssl-dev \
  python3 \
  python3-dev \
  && \
  # >=19.0.2 due to https://github.com/pypa/pip/issues/6197#issuecomment-462014853
  pip3 install -U 'pip>=19.0.2'
RUN adduser -S nb
USER nb
RUN pip3 install --user --no-cache-dir \
  'Jinja2<3' \
  'aiohttp<4' \
  'aiohttp-jinja2<2' \
  'aiohttp_session<3' \
  'cryptography<3' \
  'kubernetes_asyncio<10' \
  'python-json-logger<0.2' \
  'uvloop<0.13' \
  && true
COPY notebook /notebook
COPY notebook-worker-images /notebook
EXPOSE 5000
WORKDIR /notebook
ENTRYPOINT ["python3", "notebook.py"]
