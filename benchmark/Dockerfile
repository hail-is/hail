FROM ubuntu:18.04

ENV LANG C.UTF-8


RUN apt-get update && \
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
    python3 python3-pip \
    liblapack3 && \
  rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install aiomysql


ARG HAIL_WHEEL
COPY $HAIL_WHEEL .
RUN pip3 install $HAIL_WHEEL

ARG BENCHMARK_WHEEL
COPY $BENCHMARK_WHEEL .
RUN pip3 install $BENCHMARK_WHEEL
