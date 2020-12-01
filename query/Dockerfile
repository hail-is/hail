FROM {{ service_base_image.image }}

COPY query/setup.py /query/
COPY query/query /query/query/
RUN hail-pip-install /query && rm -rf /query

COPY query/hail.jar /

EXPOSE 5000
