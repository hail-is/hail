FROM {{ service_java_run_base_image.image }}

COPY log4j.properties /
COPY hail.jar /
COPY setup.py /query/
COPY query /query/query/
RUN hail-pip-install /query && rm -rf /query

EXPOSE 5000
