FROM continuumio/miniconda
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN mkdir batch
COPY batch/batch batch/batch
COPY batch/setup.py batch/
RUN pip install ./batch
COPY environment.yml .
RUN conda env create hail-ci -f environment.yml

WORKDIR hail-ci
COPY index.html pr-build-script ci ./

RUN conda activate hail-ci

ENTRYPOINT ["python", "ci/ci.py"]
