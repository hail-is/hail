FROM jupyter/scipy-notebook:c094bb7219f9
MAINTAINER Hail Team <hail@broadinstitute.org>

USER root
RUN install_packages openjdk-8-jre-headless
USER jovyan

RUN pip install --no-cache-dir \
  jupyter \
  jupyter-spark \
  "tornado<6" \
  hail==0.2.54 \
  jupyter_contrib_nbextensions \
  && \
  jupyter serverextension enable --user --py jupyter_spark && \
  jupyter nbextension install --user --py jupyter_spark && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable --user --py jupyter_spark && \
  jupyter nbextension enable --user --py widgetsnbextension && \
  jupyter nbextension enable --user collapsible_headings/main && \
  jupyter nbextension enable --user move_selected_cells/main

COPY ./resources/ /home/jovyan
