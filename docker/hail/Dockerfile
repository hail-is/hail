FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV MAKEFLAGS -j2

COPY hail_pip_version .

RUN apt-get update && \
  apt-get -y install \
    curl \
    git \
    liblapack3 \
    openjdk-8-jre-headless \
    python3 python3-pip \
    rsync \
    unzip bzip2 zip tar \
    vim \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 --no-cache-dir install \
      hail==$(cat hail_pip_version) \
      ipython \
      numpy \
      scipy \
      scikit-learn \
      matplotlib \
      pandas
RUN curl https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.0.1.jar \
         > $(find_spark_home.py)/jars/gcs-connector-hadoop2-2.0.1.jar && \
    mkdir -p $(find_spark_home.py)/conf && \
    touch $(find_spark_home.py)/spark-defaults.conf && \
    sed -i $(find_spark_home.py)/spark-defaults.conf \
        's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.enable.*:spark.hadoop.google.cloud.auth.service.account.enable true' \
        's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile.*:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile /gsa-key/key.json'
# source: https://cloud.google.com/storage/docs/gsutil_install#linux
RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash' && \
    mv /root/google-cloud-sdk && \
