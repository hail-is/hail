ARG BASE_IMAGE={{ hail_ubuntu_image.image }}
FROM $BASE_IMAGE

ENV VEP_VERSION 95
ENV VEP_DIR /vep

RUN apt-get update && apt-get -y install \
    build-essential \
    cpanminus \
    curl \
    git \
    libbz2-dev \
    liblzma-dev \
    libmysqlclient-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libpng-dev \
    libsqlite3-dev \
    libssl-dev \
    locales \
    mysql-client \
    openssl \
    perl \
    perl-base \
    sqlite3 \
    tabix \
    unzip \
    vim \
    wget

RUN cpanm DBI \
    DBD::mysql \
    DBD::SQLite \
    Archive::Zip \
    PerlIO::gzip \
    Set::IntervalTree \
    JSON || { for i in /root/.cpanm/work/*/build.log; do echo $i; cat $i; done }

RUN mkdir ${VEP_DIR}
WORKDIR ${VEP_DIR}

# We need to remove the original MaxEntScan.pm file from the plugins directory
# because LOFTEE has its own MaxEntScan plugin that is in directory form and VEP
# will try and use the wrong plugin.
RUN git clone -b release/95.1 https://github.com/Ensembl/ensembl-vep.git && \
    cd ensembl-vep && \
    perl INSTALL.pl -a a --NO_UPDATE && \
    echo "#!/bin/bash" > /vep/vep && \
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

COPY vep.py /hail-vep/
