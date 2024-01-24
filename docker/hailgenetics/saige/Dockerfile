# syntax=docker/dockerfile:1.3.0-labs
# 1;95;0c# ^ necessary to use an ARG in a --mount value with our current version of buildkit

ARG BASE_IMAGE
FROM $BASE_IMAGE

RUN hail-apt-get-install \
    build-essential \
    cmake \
    git \
    libfribidi-dev \
    libharfbuzz-dev \
    liblapack3 \
    libopenblas-base \
    openjdk-8-jre-headless \
    python3-pip \
    r-cran-devtools

RUN git clone --depth 1 -b main https://github.com/saigegit/SAIGE

WORKDIR /SAIGE/

ENV OMP_NUM_THREADS=1

RUN Rscript extdata/install_packages.R

RUN hail-pip-install cget pycairo

RUN R CMD INSTALL .

RUN mv extdata/step1_fitNULLGLMM.R extdata/step2_SPAtests.R extdata/step3_LDmat.R extdata/createSparseGRM.R /usr/local/bin/

RUN chmod a+x /usr/local/bin/step1_fitNULLGLMM.R && \
    chmod a+x /usr/local/bin/step2_SPAtests.R && \
    chmod a+x /usr/local/bin/step3_LDmat.R && \
    chmod a+x /usr/local/bin/createSparseGRM.R

RUN createSparseGRM.R  --help && \
    step1_fitNULLGLMM.R --help && \
    step2_SPAtests.R --help && \
    step3_LDmat.R --help

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

# ARG HAIL_WHEEL_DIR=hail/build/deploy/dist
#RUN --mount=src=${HAIL_WHEEL_DIR},target=/wheel \
#    hail-pip-install --no-deps /wheel/hail-*-py3-none-any.whl

RUN --mount=src=/wheel,target=/wheel \
    hail-pip-install --no-deps /wheel/hail-*-py3-none-any.whl
