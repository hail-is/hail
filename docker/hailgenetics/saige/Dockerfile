FROM "us-docker.pkg.dev/hail-vdc/hail/hail-ubuntu:cache-jigold"
# {{ hail_ubuntu_image.image }}

RUN apt-get update && apt-get -y install \
    git \
    wget
#    build-essential \
#    cmake \
#    libcurl4-openssl-dev \
#    libfontconfig1-dev \
#    libfreetype6-dev \
#    libfribidi-dev \
#    libharfbuzz-dev \
#    libjpeg-dev \
#    libmariadbclient-dev \
#    libmariadb-client-lgpl-dev \
#    libpng-dev \
#    libssl-dev \
#    libtiff5-dev \
#    libxml2-dev \
#    r-base-dev

RUN mkdir -p /miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    rm -rf /miniconda3/miniconda.sh

ENV PATH=/miniconda3/bin:$PATH

RUN git clone --depth 1 -b main https://github.com/saigegit/SAIGE

WORKDIR /SAIGE/conda_env/

RUN conda env create -f environment-RSAIGE.yml

#RUN conda init &&echo "conda activate RSAIGE" > ~/.bashrc

RUN conda init

SHELL ["bash", "-lc"]

RUN echo "conda activate RSAIGE" >> ~/.bashrc
    
RUN FLAGPATH=`which python | sed 's|/bin/python$||'` && \
    echo 'export LDFLAGS="-L${FLAGPATH}/lib"' >> ~/.bash_profile && \
    echo 'export CPPFLAGS="-I${FLAGPATH}/include"' >> ~/.bash_profile


#RUN git clone --depth 1 -b main https://github.com/saigegit/SAIGE

# RUN Rscript ./SAIGE/extdata/install_packages.R

# RUN R CMD INSTALL SAIGE
