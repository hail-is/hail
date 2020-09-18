FROM bitnami/minideb:buster

ENV LANG C.UTF-8


RUN echo 'deb http://ftp.us.debian.org/debian stretch main' >> /etc/apt/sources.list && \
  install_packages \
    git \
    htop \
    unzip bzip2 zip tar \
    wget curl \
    rsync \
    emacs25-nox \
    default-mysql-client \
    xsltproc pandoc \
    jq \
    openjdk-8-jdk-headless \
    python \
    python3.7 python3-pip python3.7-dev \
    liblapack3 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1


COPY requirements.txt .
RUN pip3 -q install -r requirements.txt

RUN pip3 -q install aiomysql


ARG HAIL_WHEEL
COPY $HAIL_WHEEL .
RUN pip3 -q install $HAIL_WHEEL

ARG BENCHMARK_WHEEL
COPY $BENCHMARK_WHEEL .
RUN pip3 -q install $BENCHMARK_WHEEL
