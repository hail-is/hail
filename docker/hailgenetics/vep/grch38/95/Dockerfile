ARG DOCKER_PREFIX={{ global.docker_prefix }}
FROM $DOCKER_PREFIX/ubuntu:22.04

ENV VEP_VERSION=95
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
    libarchive-zip-perl \
    libdbd-mysql-perl \
    libdbd-sqlite3-perl \
    libdbi-perl \
    libipc-run-perl \
    libjson-perl \
    libperlio-gzip-perl \
    libset-intervaltree-perl \
    libxml-libxml-perl

RUN mkdir ${VEP_DIR}
WORKDIR ${VEP_DIR}

# We need to remove the original MaxEntScan.pm file from the plugins directory
# because LOFTEE has its own MaxEntScan plugin that is in directory form and VEP
# will try and use the wrong plugin.
RUN git clone -b release/95.1 https://github.com/Ensembl/ensembl-vep.git && \
    cd ensembl-vep && \
    perl INSTALL.pl -a a --NO_UPDATE && \
    echo '#!/bin/bash' > /vep/vep && \
    echo "export PERL5LIB=\$PERL5LIB:${VEP_DIR}/ensembl-vep/Plugins/" >> /vep/vep && \
    echo "exec perl ${VEP_DIR}/ensembl-vep/vep \"\$@\"" >> /vep/vep && \
    chmod +x /vep/vep && \
    git clone -b postreleasefix/95 https://github.com/Ensembl/VEP_plugins.git Plugins/ && \
    git clone -b grch38 https://github.com/konradjk/loftee.git && \
    mv loftee/* Plugins/ && \
    rm Plugins/MaxEntScan.pm

RUN export KENT_SRC=$PWD/kent-335_base/src && \
    export MACHTYPE=$(uname -m) && \
    export CFLAGS="-fPIC" && \
    export MYSQLINC="mysql_config --include | sed -e 's/^-I//g'" && \
    export MYSQLLIBS="mysql_config --libs" && \
    wget https://github.com/ucscGenomeBrowser/kent/archive/v335_base.tar.gz && \
    tar xzf v335_base.tar.gz && \
    cd $KENT_SRC/lib && \
    echo 'CFLAGS="-fPIC"' > ../inc/localEnvironment.mk && \
    make clean && make && \
    cd ../jkOwnLib && \
    make clean && make && \
    mkdir -p $VEP_DIR/cpanm && \
    export PERL5LIB=\$PERL5LIB:$HOME/cpanm/lib/perl5 && \
    cpanm Bio::DB::BigFile

RUN wget https://github.com/samtools/samtools/releases/download/1.22/samtools-1.22.tar.bz2 && \
    bzip2 -d samtools-1.22.tar.bz2 && \
    tar -xf samtools-1.22.tar && \
    cd samtools-1.22 && \
    ./configure && \
    make && \
    make install

COPY vep.py /hail-vep/
