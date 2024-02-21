# syntax=docker/dockerfile:1.3.0-labs
# 1;95;0c# ^ necessary to use an ARG in a --mount value with our current version of buildkit

ARG BASE_IMAGE
FROM $BASE_IMAGE

ARG PYTHON_VERSION=3.9

RUN hail-apt-get-install \
    build-essential \
    cmake \
    git \
    libcairo2-dev \
    libcurl4-openssl-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    liblapack3 \
    libopenblas-base \
    libpng-dev \
    libssl-dev \
    libtiff5-dev \
    libxml2-dev \
    openjdk-8-jre-headless \
    r-base-dev \
    wget && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_VERSION 1

RUN hail-pip-install cget==0.2.0 pycairo==1.25.1

RUN git clone --depth 1 -b v1.3.3 https://github.com/saigegit/SAIGE

WORKDIR /SAIGE/

ENV OMP_NUM_THREADS=1

RUN Rscript extdata/install_packages.R

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

WORKDIR /

RUN wget https://github.com/samtools/htslib/releases/download/1.19.1/htslib-1.19.1.tar.bz2 && \
    bzip2 -d htslib-1.19.1.tar.bz2 && \
    tar -xf htslib-1.19.1.tar && \
    cd htslib-1.19.1 && \
    ./configure && \
    make && \
    make install && \
    mv tabix /usr/local/bin/ && \
    chmod a+x /usr/local/bin/tabix

#COPY hail/python/pinned-requirements.txt requirements.txt
#RUN hail-pip-install -r requirements.txt scikit-learn ipython

#RUN export SPARK_HOME=$(find_spark_home.py) && \
#    curl https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop2-2.2.7.jar \
#         >$SPARK_HOME/jars/gcs-connector-hadoop2-2.2.7.jar && \
#    mkdir -p $SPARK_HOME/conf && \
#    touch $SPARK_HOME/conf/spark-defaults.conf && \
#    sed -i $SPARK_HOME/conf/spark-defaults.conf \
#        -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.enable.*:spark.hadoop.google.cloud.auth.service.account.enable # true:' \
#        -e 's:spark\.hadoop\.google\.cloud\.auth\.service\.account\.json\.keyfile.*:spark\.hadoop\.google\.cloud\.auth\.service\.ac# count\.json\.keyfile /gsa-key/key.json:'

#ARG HAIL_WHEEL_DIR=hail/build/deploy/dist
#RUN --mount=src=${HAIL_WHEEL_DIR},target=/wheel \
#    hail-pip-install --no-deps /wheel/hail-*-py3-none-any.whl
