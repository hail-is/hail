FROM continuumio/miniconda
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN mkdir /home/hail-ci && \
    groupadd hail-ci && \
    useradd -g hail-ci hail-ci && \
    chown hail-ci:hail-ci /home/hail-ci

WORKDIR /batch
COPY batch/batch batch
COPY batch/setup.py .

WORKDIR /hail-ci
COPY ci/environment.yml .
RUN conda env create -n hail-ci -f environment.yml && \
    rm -rf /opt/conda/pkgs/*

COPY ci/index.html ci/pr-build-script ci/pr-deploy-script ci/deploy-index.html ./
COPY ci/ci ./ci
COPY ci/run_ci.py ./
RUN chown -R hail-ci:hail-ci ./

USER hail-ci
ENV PATH /opt/conda/envs/hail-ci/bin:$PATH
RUN pip install --user /batch
EXPOSE 5000
VOLUME /hail-ci/oauth-token
VOLUME /hail-ci/gcloud-token
ENTRYPOINT ["python", "run_ci.py"]
