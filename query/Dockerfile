FROM {{ service_base_image.image }}

COPY query/setup.py /query/
COPY query/query /query/query/
RUN python3 -m hailpip --no-cache-dir /query && \
  rm -rf /query

COPY query/hail.jar /

EXPOSE 5000
