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
COPY environment.yml .
RUN conda env create -n hail-ci -f environment.yml && \
    rm -rf /opt/conda/pkgs/*

COPY index.html pr-build-script pr-deploy-script ./
COPY ci ./ci
RUN chown -R hail-ci:hail-ci ./

USER hail-ci
ENV PATH /opt/conda/envs/hail-ci/bin:$PATH
RUN pip install --user /batch
EXPOSE 5000
VOLUME /hail-ci/oauth-token
VOLUME /hail-ci/gcloud-token
ENTRYPOINT ["python", "ci/ci.py"]
