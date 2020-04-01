FROM {{ service_base_image.image }}

COPY query/setup.py batch/MANIFEST.in /query/
COPY query/query /query/query/
RUN pip3 install --no-cache-dir /query && \
  rm -rf /query

EXPOSE 5000
