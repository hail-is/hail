FROM {{ service_java_run_base_image.image }}

COPY query/log4j.properties /
COPY query/hail.jar /
COPY query/setup.py /query/
COPY query/query /query/query/
RUN hail-pip-install /query && rm -rf /query

EXPOSE 5000
