FROM {{ service_base_image.image }}

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/
RUN python3 -m hailpip --no-cache-dir /batch && \
  rm -rf /batch

EXPOSE 5000
