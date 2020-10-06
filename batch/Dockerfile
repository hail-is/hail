FROM {{ service_base_image.image }}

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/
RUN pip3 install --use-feature=2020-resolver --upgrade --no-cache-dir /batch && \
  rm -rf /batch && \
  python3 -m pip check

EXPOSE 5000
