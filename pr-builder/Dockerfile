FROM continuumio/miniconda
MAINTAINER Hail Team <hail@broadinstitute.org>

USER root

RUN apt-get update && apt-get -y install \
    bash \
    cmake \
    curl \
    g++ \
    git \
    libcurl4-openssl-dev \
    libnlopt-dev \
    libxml2-dev \
    openjdk-8-jdk-headless \
    r-base \
    r-cran-ncdf4 \
    r-cran-rnetcdf \
    tar \
    unzip \
    xsltproc \
    pandoc \
    && apt-get clean

RUN useradd --create-home --shell /bin/bash hail

RUN Rscript -e 'install.packages(c("jsonlite", "SKAT", "logistf"), repos = "http://cran.us.r-project.org")' && \
    Rscript -e 'source("https://bioconductor.org/biocLite.R"); biocLite("GENESIS"); biocLite("SNPRelate"); biocLite("GWASTools")'

WORKDIR /plink
ADD http://www.cog-genomics.org/static/bin/plink/plink_linux_x86_64.zip /plink
RUN unzip plink_linux_x86_64.zip && \
    rm -rf plink_linux_x86_64.zip && \
    ln -s /plink/plink /bin/plink

WORKDIR /qctool
ADD http://www.well.ox.ac.uk/~gav/resources/qctool_v2.0.1-CentOS6.8-x86_64.tgz /qctool
RUN tar --strip-components=1 -xvzf qctool_v2.0.1-CentOS6.8-x86_64.tgz -C /qctool && \
    rm -rf qctool_v2.0.1-CentOS6.8-x86_64.tgz && \
    ln -s /qctool/qctool /bin/qctool

WORKDIR /spark
ADD https://archive.apache.org/dist/spark/spark-2.0.2/spark-2.0.2-bin-hadoop2.7.tgz /spark
RUN tar --strip-components=1 -xvzf spark-2.0.2-bin-hadoop2.7.tgz -C /spark && \
    rm -rf spark-2.0.2-bin-hadoop2.7.tgz
ENV SPARK_HOME /spark/

# this seems easier than getting the keys right for apt
#
# source: https://cloud.google.com/storage/docs/gsutil_install#linux
RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash'
ENV PATH $PATH:/root/google-cloud-sdk/bin
# gcloud iam key files will be stored here
VOLUME /secrets

# cache (almost all) gradle dependencies (and gradle itself)
#
# Some plugin we use hides its dependencies in a so-called "detached
# configuration" so there is no way for us to ensure it gets cached.
WORKDIR /hail
COPY gradlew build.gradle settings.gradle ./
COPY gradle gradle
RUN mkdir -p /gradle-cache
RUN ./gradlew downloadDependencies --gradle-user-home /gradle-cache

COPY dev-environment.yml dev-environment.yml
RUN conda env create -f ./dev-environment.yml && \
    rm -rf /opt/conda/pkgs/*

RUN chown -R hail:hail /hail /gradle-cache

USER hail
