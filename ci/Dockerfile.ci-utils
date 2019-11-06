FROM {{ base_image.image }}

RUN apt-get update && \
  apt-get -y install docker.io && \
  rm -rf /var/lib/apt/lists/*

COPY jinja2_render.py .
COPY wait-for.py .
