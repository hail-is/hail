ARG BASE_IMAGE={{ hail_ubuntu_image.image }}
FROM $BASE_IMAGE

RUN curl -sLO https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.3/tailwindcss-linux-x64 && \
    chmod +x tailwindcss-linux-x64 && \
    mv tailwindcss-linux-x64 /usr/bin/tailwindcss

COPY hail/python/hailtop/pinned-requirements.txt hailtop-requirements.txt
COPY gear/pinned-requirements.txt gear-requirements.txt
COPY web_common/pinned-requirements.txt web_common-requirements.txt
COPY batch/pinned-requirements.txt batch-requirements.txt
RUN hail-pip-install \
      -r hailtop-requirements.txt \
      -r gear-requirements.txt \
      -r web_common-requirements.txt \
      -r batch-requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/

COPY gear/pyproject.toml /gear/pyproject.toml
COPY gear/gear /gear/gear/

COPY web_common/pyproject.toml web_common/MANIFEST.in web_common/input.css web_common/tailwind.config.js /web_common/
COPY web_common/web_common /web_common/web_common/

COPY batch/pyproject.toml /batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/

RUN cd web_common && \
    mkdir web_common/static/css && \
    tailwindcss -i input.css -o web_common/static/css/output.css

RUN hail-pip-install /hailtop /gear /web_common /batch

EXPOSE 5000
