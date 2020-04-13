FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV MAKEFLAGS -j2

# COPY wheel-container.tar .

# RUN apt-get update && \
#   apt-get -y install \
#     curl \
#     git \
#     liblapack3 \
#     openjdk-8-jre-headless \
#     python3 python3-pip \
#     rsync \
#     unzip bzip2 zip tar \
#     vim \
#   && rm -rf /var/lib/apt/lists/*
# RUN tar xvf wheel-container.tar && \
#     pip3 --no-cache-dir install \
#       hail-*-py3-none-any.whl \
#       ipython \
#       matplotlib \
#       numpy \
#       pandas \
#       scikit-learn \
#       scipy \
#     && rm -rf hail-*-py3-none-any.whl
# RUN curl https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.0.1.jar \
#          > $(find_spark_home.py)/jars/gcs-connector-hadoop2-2.0.1.jar && \
#     mkdir -p $(find_spark_home.py)/conf && \
#     touch $(find_spark_home.py)/spark-defaults.conf && \
#     sed -i $(find_spark_home.py)/spark-defaults.conf \
#         -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.enable.*:spark.hadoop.google.cloud.auth.service.account.enable true:' \
#         -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile.*:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile /gsa-key/key.json:'
# # source: https://cloud.google.com/storage/docs/gsutil_install#linux
# RUN curl https://sdk.cloud.google.com | bash && \
#     echo 'source /root/google-cloud-sdk/completion.bash.inc' >> ~/.profile && \
#     echo 'source /root/google-cloud-sdk/path.bash.inc' >> ~/.profile
