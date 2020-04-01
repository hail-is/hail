FROM {{ service_base_image.image }}

COPY query/setup.py /query/
COPY query/query /query/query/
RUN pip3 install --no-cache-dir /query && \
  rm -rf /query

COPY hail/build/libs/hail-all-spark.jar /

EXPOSE 5000
