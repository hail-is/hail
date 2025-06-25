ARG DOCKER_PREFIX={{ global.docker_prefix }}
FROM $DOCKER_PREFIX/ubuntu:22.04

ENV VEP_VERSION=85
ENV VEP_DIR=/vep

RUN apt-get update && apt-get -y install \
    build-essential \
    cpanminus \
    curl \
    git \
    locales \
    mysql-client \
    openssl \
    perl \
    perl-base \
    python3 \
    sqlite3 \
    tabix \
    unzip \
    wget \
    libbz2-dev \
    libcurl4-openssl-dev \
    liblzma-dev \
    libmysqlclient-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libpng-dev \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    libarchive-extract-perl \
    libarchive-zip-perl \
    libbio-db-hts-perl \
    libdbd-mysql-perl \
    libdbd-sqlite3-perl \
    libdbi-perl \
    libdevel-checklib-perl \
    libfile-copy-recursive-perl \
    libjson-perl

RUN mkdir $VEP_DIR
WORKDIR $VEP_DIR

RUN wget https://github.com/Ensembl/ensembl-tools/archive/release/${VEP_VERSION}.zip && \
    unzip ${VEP_VERSION}.zip && \
    cd ensembl-tools-release-${VEP_VERSION}/scripts/variant_effect_predictor/ && \
    perl INSTALL.pl -a a --NO_HTSLIB && \
    echo '#!/bin/bash' > /vep/vep && \
    echo "export PERL5LIB=\$PERL5LIB:${VEP_DIR}/loftee" >> /vep/vep && \
    echo "exec perl ${VEP_DIR}/ensembl-tools-release-${VEP_VERSION}/scripts/variant_effect_predictor/variant_effect_predictor.pl \"\$@\"" >> /vep/vep && \
    chmod +x /vep/vep

RUN git clone -b v1.0.2 https://github.com/konradjk/loftee.git

RUN wget https://github.com/samtools/samtools/releases/download/1.22/samtools-1.22.tar.bz2 && \
    bzip2 -d samtools-1.22.tar.bz2 && \
    tar -xf samtools-1.22.tar && \
    cd samtools-1.22 && \
    ./configure && \
    make && \
    make install

COPY vep.py /hail-vep/
