FROM "us-docker.pkg.dev/hail-vdc/hail/hail-ubuntu:cache-jigold"
# {{ hail_ubuntu_image.image }}

RUN apt-get update && apt-get -y install \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    libmariadbclient-dev \
    libmariadb-client-lgpl-dev \
    libpng-dev \
    libssl-dev \
    libtiff5-dev \
    libxml2-dev \
    r-base-dev

RUN git clone --depth 1 -b main https://github.com/saigegit/SAIGE

# RUN Rscript ./SAIGE/extdata/install_packages.R

# RUN R CMD INSTALL SAIGE
