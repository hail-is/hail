FROM {{ hail_public_image.image }}

RUN apt-get update && \
  apt-get -y install \
    cmake \
    libbz2-dev \
    libcurl4-openssl-dev \
    liblzma-dev \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir samtools && \
    (cd samtools && \
     curl -LO https://github.com/samtools/samtools/releases/download/1.9/samtools-1.9.tar.bz2 && \
     tar -xf samtools-1.9.tar.bz2 && \
     rm -rf samtools-1.9.tar.bz2 && \
     cd samtools-1.9 && \
     ./configure --without-curses && \
     make && \
     make install)
RUN git clone --recursive https://github.com/vcflib/vcflib.git && \
    (cd vcflib && make)
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
     curl -LO http://people.virginia.edu/~wc9c/KING/Linux-king.tar.gz && \
     tar -xf Linux-king.tar.gz)
RUN mkdir gcta && \
    (cd gcta && \
     curl -LO https://cnsgenomics.com/software/gcta/bin/gcta_1.93.1beta.zip && \
     unzip gcta_1.93.1beta.zip && \
     rm -rf gcta_1.93.1beta.zip)

RUN apt-get update && \
  apt-get -y install \
    libgsl-dev \
    liblapacke-dev \
    libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir eigenstrat && \
    (cd eigenstrat && \
     git clone https://github.com/DReichLab/EIG.git && \
     cd EIG && \
     cd src && \
     LDLIBS="-llapacke" make && \
     make install)
RUN find \
      eigenstrat/EIG/bin \
      gcta/gcta_1.91.7beta \
      king \
      -type f -executable \
    | xargs -I % /bin/sh -c 'set -ex ; ln -s ${PWD}/% /usr/local/bin/$(basename %)'
