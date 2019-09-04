FROM ubuntu:18.04

ENV LANG C.UTF-8

ARG WHEEL

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
    python3 python3-pip && \
  rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY $WHEEL .
RUN pip3 install $WHEEL