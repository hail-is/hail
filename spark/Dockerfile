FROM debian:9.5

RUN apt-get update && \
  apt-get -y install \
    wget && \
  rm -rf /var/lib/apt/lists/*

RUN wget -O spark-2.2.0-bin-hadoop2.7.tgz https://archive.apache.org/dist/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz && \
  tar xzf spark-2.2.0-bin-hadoop2.7.tgz && \
  rm spark-2.2.0-bin-hadoop2.7.tgz

RUN apt-get update && \
  apt-get -y install \
    procps \
    unzip \
    bzip2 \
    default-jdk && \
  rm -rf /var/lib/apt/lists/*
