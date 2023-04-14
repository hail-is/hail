FROM {{ hail_ubuntu_image.image }}

ENV VEP_VERSION 85
ENV VEP_DIR /vep

RUN apt-get update && apt-get -y install \
    apache2 \
    build-essential \
    cpanminus \
    curl \
    git \
    libmysqlclient-dev \
    libpng-dev \
    libssl-dev \
    locales \
    manpages \
    mysql-client \
    openssl \
    perl \
    perl-base \
    samtools \
    sqlite3 \
    tabix \
    unzip \
    vim \
    wget

RUN cpanm Devel::CheckLib \
    DBI \
    DBD::mysql \
    DBD::SQLite \
    Try \
    File::Copy::Recursive \
    Archive::Extract \
    Archive::Zip \
    CGI \
    Bio::Root::Version \
    Bio::DB:HTS \
    JSON || { for i in /root/.cpanm/work/*/build.log; do echo $i; cat $i; done }

RUN mkdir $VEP_DIR
WORKDIR $VEP_DIR

RUN wget https://github.com/Ensembl/ensembl-tools/archive/release/${VEP_VERSION}.zip && \
    unzip ${VEP_VERSION}.zip && \
    cd ensembl-tools-release-${VEP_VERSION}/scripts/variant_effect_predictor/ && \
    perl INSTALL.pl -a a --NO_HTSLIB && \
    echo "#!/bin/bash" > /vep/vep && \
    echo "export PERL5LIB=\$PERL5LIB:${VEP_DIR}/loftee" >> /vep/vep && \
    echo "exec perl ${VEP_DIR}/ensembl-tools-release-${VEP_VERSION}/scripts/variant_effect_predictor/variant_effect_predictor.pl \"\$@\"" >> /vep/vep && \
    chmod +x /vep/vep

RUN git clone -b v1.0.2 https://github.com/konradjk/loftee.git

COPY vep.py /hail-vep/
