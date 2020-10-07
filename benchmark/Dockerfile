FROM ubuntu:18.04

ENV LANG C.UTF-8

RUN echo "APT::Acquire::Retries \"3\";" > /etc/apt/apt.conf.d/80-retries && \
  apt-get update && \
  apt-get -y install \
    git \
    htop \
    unzip bzip2 zip tar \
    wget curl \
    rsync \
    emacs25-nox \
    mysql-client \
    xsltproc pandoc \
    jq \
    openjdk-8-jdk-headless \
    python \
    python3.7 python3-pip python3.7-dev \
    liblapack3 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
  rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir --upgrade pip && \
  python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir -r requirements.txt && \
  python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir aiomysql && \
  python3 -m pip check

ARG HAIL_WHEEL
COPY $HAIL_WHEEL .
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir --quiet $HAIL_WHEEL && \
  python3 -m pip check

ARG BENCHMARK_WHEEL
COPY $BENCHMARK_WHEEL .
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir --quiet $BENCHMARK_WHEEL && \
  python3 -m pip check
