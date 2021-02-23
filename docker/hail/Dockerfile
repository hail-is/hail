FROM {{ hail_ubuntu_image.image }}

ENV MAKEFLAGS -j2

RUN hail-apt-get-install \
    curl \
    git \
    liblapack3 \
    openjdk-8-jre-headless \
    python3 python3-pip \
    rsync \
    unzip bzip2 zip tar \
    vim
COPY hail_pip_version /
RUN hail-pip-install \
      hail==$(cat /hail_pip_version) \
      ipython \
      matplotlib \
      numpy \
      pandas \
      scikit-learn \
      scipy \
    && rm -rf hail-*-py3-none-any.whl
RUN export SPARK_HOME=$(find_spark_home.py) && \
    curl https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.0.1.jar \
         >$SPARK_HOME/jars/gcs-connector-hadoop2-2.0.1.jar && \
    mkdir -p $SPARK_HOME/conf && \
    touch $SPARK_HOME/conf/spark-defaults.conf && \
    sed -i $SPARK_HOME/conf/spark-defaults.conf \
        -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.enable.*:spark.hadoop.google.cloud.auth.service.account.enable true:' \
        -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile.*:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile /gsa-key/key.json:'
# source: https://cloud.google.com/storage/docs/gsutil_install#linux
RUN curl https://sdk.cloud.google.com | bash && \
    find \
      /root/google-cloud-sdk/bin/ \
      -type f -executable \
    | xargs -I % /bin/sh -c 'set -ex ; ln -s ${PWD}/% /usr/local/bin/$(basename %)'
