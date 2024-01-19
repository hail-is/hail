FROM {{ hail_ubuntu_image.image }}

RUN apt-get update && apt-get -y install \
    build-essential \
    cmake \
    git \
    r-base \
    r-base-dev

RUN git clone --depth 1 -b main https://github.com/saigegit/SAIGE

RUN Rscript ./SAIGE/extdata/install_packages.R

RUN R CMD INSTALL SAIGE
