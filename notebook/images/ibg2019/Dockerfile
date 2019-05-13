FROM jupyter/scipy-notebook
MAINTAINER Hail Team <hail@broadinstitute.org>

USER root
RUN apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*
USER jovyan

RUN pip install --no-cache-dir \
  'jupyter-spark<0.5' \
  hail==0.2.11 \
  jupyter_contrib_nbextensions \
  && \
  jupyter serverextension enable --user --py jupyter_spark && \
  jupyter nbextension install --user --py jupyter_spark && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable --user --py jupyter_spark && \
  jupyter nbextension enable --user --py widgetsnbextension && \
  jupyter nbextension enable --user collapsible_headings/main && \
  jupyter nbextension enable --user move_selected_cells/main

RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash' && \
    ./google-cloud-sdk/bin/gcloud components install beta
ENV PATH $PATH:/home/jovyan/google-cloud-sdk/bin

RUN rm -r work/

RUN gsutil -m cp -r "gs://hail-tutorial/ibg2019/*" /home/jovyan/.