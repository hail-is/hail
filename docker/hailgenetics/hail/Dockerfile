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

RUN --mount=src=wheel-container.tar,target=/wheel-container.tar \
    tar -xf wheel-container.tar && \
    hail-pip-install --no-deps hail-*-py3-none-any.whl && \
    rm -rf hail-*-py3-none-any.whl
