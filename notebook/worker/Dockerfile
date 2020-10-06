FROM jupyter/scipy-notebook
MAINTAINER Hail Team <hail@broadinstitute.org>

USER root
RUN apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*
USER jovyan

RUN pip install --use-feature=2020-resolver --upgrade --no-cache-dir \
  'jupyter-spark<0.5' \
  hail==0.2.4 \
  jupyter_contrib_nbextensions \
  && \
  pip check && \
  jupyter serverextension enable --user --py jupyter_spark && \
  jupyter nbextension install --user --py jupyter_spark && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable --user --py jupyter_spark && \
  jupyter nbextension enable --user --py widgetsnbextension && \
  jupyter nbextension enable --user collapsible_headings/main && \
  jupyter nbextension enable --user move_selected_cells/main

COPY ./resources/ /home/jovyan
