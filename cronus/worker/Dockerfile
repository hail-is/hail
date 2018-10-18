FROM jupyter/scipy-notebook
MAINTAINER Hail Team <hail@broadinstitute.org>

USER root
RUN apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*
USER jovyan

RUN pip install --no-cache-dir hail==0.2.2
