FROM {{ service_base_image.image }}

COPY query/setup.py /query/
COPY query/query /query/query/
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /query && \
  rm -rf /query && \
  python3 -m pip check

COPY query/hail.jar /

EXPOSE 5000
