FROM {{ hailgenetics_hail_image.image }}

RUN hail-apt-get-install \
    cmake \
    libbz2-dev \
    libcurl4-openssl-dev \
    liblzma-dev \
    libvcflib-tools \
    libvcflib-dev \
    zlib1g-dev

RUN mkdir samtools && \
    (cd samtools && \
     curl -LO https://github.com/samtools/samtools/releases/download/1.9/samtools-1.9.tar.bz2 && \
     tar -xf samtools-1.9.tar.bz2 && \
     rm -rf samtools-1.9.tar.bz2 && \
     cd samtools-1.9 && \
     ./configure --without-curses && \
     make && \
     make install)
RUN curl -LO http://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_latest.zip && \
    unzip plink2_linux_avx2_latest.zip && \
    mv plink2 /bin/ && \
    rm -rf plink2_linux_avx2_latest.zip
RUN curl -LO http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_latest.zip && \
    unzip plink_linux_x86_64_latest.zip && \
    mv plink /bin/ && \
    rm -rf plink_linux_x86_64_latest.zip
RUN mkdir king && \
    (cd king && \
     curl -LO https://www.kingrelatedness.com/Linux-king.tar.gz && \
     tar -xf Linux-king.tar.gz --no-same-owner --no-same-permissions)
RUN mkdir gcta && \
    (cd gcta && \
     curl -LO https://cnsgenomics.com/software/gcta/bin/gcta_1.93.1beta.zip && \
     unzip gcta_1.93.1beta.zip && \
     rm -rf gcta_1.93.1beta.zip)

RUN hail-apt-get-install \
    libgsl-dev \
    liblapacke-dev \
    libopenblas-dev

RUN mkdir eigenstrat && \
    (cd eigenstrat && \
     git clone https://github.com/DReichLab/EIG.git && \
     cd EIG && \
     git checkout v7.2.1 && \
     cd src && \
     LDLIBS="-llapacke" make && \
     make install)
RUN find \
      eigenstrat/EIG/bin \
      gcta/gcta_1.91.7beta \
      king \
      -type f -executable \
    | xargs -I % /bin/sh -c 'set -ex ; ln -s ${PWD}/% /usr/local/bin/$(basename %)'
