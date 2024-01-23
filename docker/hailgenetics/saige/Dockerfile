FROM {{ hail_run_image.image }}

RUN apt-get update && apt-get -y install \
    build-essential \
    cmake \
    git \
    libfribidi-dev \
    libharfbuzz-dev \
    libopenblas-base \
    python3-pip \
    r-cran-devtools

RUN git clone --depth 1 -b main https://github.com/saigegit/SAIGE

WORKDIR /SAIGE/

ENV OMP_NUM_THREADS=1

RUN Rscript extdata/install_packages.R

RUN pip3 install cget

RUN R CMD INSTALL .

RUN mv extdata/step1_fitNULLGLMM.R extdata/step2_SPAtests.R extdata/step3_LDmat.R extdata/createSparseGRM.R /usr/local/bin/

RUN chmod a+x /usr/local/bin/step1_fitNULLGLMM.R && \
    chmod a+x /usr/local/bin/step2_SPAtests.R && \
    chmod a+x /usr/local/bin/step3_LDmat.R && \
    chmod a+x /usr/local/bin/createSparseGRM.R

RUN createSparseGRM.R  --help && \
    step1_fitNULLGLMM.R --help && \
    step2_SPAtests.R --help && \
    step3_LDmat.R --help