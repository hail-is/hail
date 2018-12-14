FROM jupyter/scipy-notebook
MAINTAINER Hail Team <hail@broadinstitute.org>

USER root
ENV MAKE="make -j8"
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    g++ \
    gcc \
    git \
    less \
    libgsl-dev \
    liblapacke-dev \
    libopenblas-dev \
    openjdk-8-jre-headless \
    perl \
    r-base \
    && \
    rm -rf /var/lib/apt/lists/* \
    && \
    Rscript -e "install.packages(c('IRkernel', 'ggplot2'))"
USER jovyan

RUN pip install --no-cache-dir \
  'jupyter-spark<0.5' \
  hail==0.2.4 \
  numpy==1.15.4 \
  pandas==0.23.4 \
  matplotlib==3.0.2 \
  scikit-learn \
  jupyter_contrib_nbextensions \
  && \
  Rscript -e 'IRkernel::installspec()' \
  && \
  jupyter serverextension enable --user --py jupyter_spark && \
  jupyter nbextension install --user --py jupyter_spark && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable --user --py jupyter_spark && \
  jupyter nbextension enable --user --py widgetsnbextension && \
  jupyter nbextension enable --user collapsible_headings/main && \
  jupyter nbextension enable --user move_selected_cells/main

RUN mkdir samtools && \
    (cd samtools && \
     wget https://github.com/samtools/samtools/releases/download/1.9/samtools-1.9.tar.bz2 && \
     tar -xf samtools-1.9.tar.bz2 && \
     rm -rf samtools-1.9.tar.bz2 && \
     cd samtools-1.9 && \
     ./configure && \
     make -j8)

RUN git clone --recursive https://github.com/vcflib/vcflib.git && \
    (cd vcflib && make -j8)

RUN mkdir vcftools && \
    (cd vcftools && \
     wget https://github.com/vcftools/vcftools/releases/download/v0.1.16/vcftools-0.1.16.tar.gz && \
     tar -xf vcftools-0.1.16.tar.gz && \
     rm -rf vcftools-0.1.16.tar.gz && \
     cd vcftools-0.1.16 && \
     ./configure && \
     make -j8) && \
     echo 'export PERL5LIB='${HOME}'/vcftools/src/perl/' >> ${HOME}/.profile

RUN mkdir plink && \
    (cd plink && \
     wget http://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_20181028.zip && \
     unzip plink2_linux_x86_64_20181028.zip && \
     rm -rf plink2_linux_x86_64_20181028.zip)

RUN mkdir king && \
    (cd king && \
     wget http://people.virginia.edu/~wc9c/KING/Linux-king.tar.gz && \
     tar -xf Linux-king.tar.gz)

RUN mkdir gcta && \
    (cd gcta && \
     wget http://cnsgenomics.com/software/gcta/gcta_1.91.7beta.zip && \
     unzip gcta_1.91.7beta.zip && \
     rm -rf gcta_1.91.7beta.zip)

RUN mkdir eigenstrat && \
    (cd eigenstrat && \
     git clone https://github.com/DReichLab/EIG.git && \
     cd EIG && \
     cd src && \
     LDLIBS="-llapacke" make -j8 && \
     make install)

RUN mkdir plink19 && \
    (cd plink19 && \
     wget http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20181202.zip && \
     unzip plink_linux_x86_64_20181202.zip && \
     rm -rf plink_linux_x86_64_20181202.zip)

USER root
RUN (cd vcftools/vcftools-0.1.16 && make install) && \
    (cd samtools/samtools-1.9 && make install) && \
    find \
         eigenstrat/EIG/bin \
         gcta/gcta_1.91.7beta \
         king \
         plink \
         plink19 \
         vcflib/bin \
         -type f -executable \
       | xargs -I % /bin/sh -c 'set -ex ; ln -s ${PWD}/% /usr/local/bin/$(basename %)'
RUN chown jovyan:users -R /home/jovyan
USER jovyan

ENV R_LIBS=/home/jovyan/.R_libs/
RUN mkdir -p $R_LIBS

RUN wget -O WEScourse.ISB101.2018.tar.gz https://www.dropbox.com/s/fwqy9vze5p7w7wi/WEScourse.ISB101.2018.tar.gz?dl=1 && \
    tar -xf WEScourse.ISB101.2018.tar.gz && \
    rm -rf WEScourse.ISB101.2018.tar.gz

COPY ccg_workshop_2018_python3.ipynb .

COPY python_part_2.ipynb .

COPY some_numbers.txt .

COPY stooges_fixed.csv .

COPY stooges.csv .

RUN Rscript -e 'source("https://bioconductor.org/biocLite.R"); \
                biocLite("Biostrings"); \
                install.packages("dplyr")'

<<<<<<< HEAD
USER root
RUN apt-get update && apt-get install -y vim dos2unix
USER jovyan

RUN mkdir genotype-harmonizer && \
    (cd genotype-harmonizer && \
     wget https://molgenis26.gcc.rug.nl/downloads/GenotypeHarmonizer/GenotypeHarmonizer-1.4.20-dist.tar.gz && \
     tar -xf GenotypeHarmonizer-1.4.20-dist.tar.gz && \
     rm -rf GenotypeHarmonizer-1.4.20-dist.tar.gz && \
     find . -name '*.sh' | xargs dos2unix)

COPY workshop_cars.txt workshop_plants.txt ./
