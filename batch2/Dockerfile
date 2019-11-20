FROM {{ service_base_image.image }}

COPY setup.py MANIFEST.in /batch/
COPY batch /batch/batch/
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

EXPOSE 5000
