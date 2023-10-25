ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN hail-apt-get-install \
    git \
    liblapack3 \
    openjdk-8-jre-headless

COPY hail/python/pinned-requirements.txt requirements.txt
RUN hail-pip-install -r requirements.txt scikit-learn ipython

RUN export SPARK_HOME=$(find_spark_home.py) && \
    curl https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.2.7.jar \
         >$SPARK_HOME/jars/gcs-connector-hadoop2-2.2.7.jar && \
    mkdir -p $SPARK_HOME/conf && \
    touch $SPARK_HOME/conf/spark-defaults.conf && \
    sed -i $SPARK_HOME/conf/spark-defaults.conf \
        -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.enable.*:spark.hadoop.google.cloud.auth.service.account.enable true:' \
        -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile.*:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile /gsa-key/key.json:'

ARG HAIL_WHEEL_DIR=hail/build/deploy/dist
RUN --mount=src=${HAIL_WHEEL_DIR},target=/wheel \
    hail-pip-install --no-deps /wheel/hail-*-py3-none-any.whl
