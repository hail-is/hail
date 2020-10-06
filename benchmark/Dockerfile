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
RUN pip3 -q install -r requirements.txt

RUN pip3 -q install aiomysql


ARG HAIL_WHEEL
COPY $HAIL_WHEEL .
RUN pip3 -q install $HAIL_WHEEL

ARG BENCHMARK_WHEEL
COPY $BENCHMARK_WHEEL .
RUN pip3 -q install $BENCHMARK_WHEEL
