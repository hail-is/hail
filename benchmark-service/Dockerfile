FROM {{ service_base_image.image }}

COPY benchmark-service/benchmark /benchmark-service/benchmark/

COPY benchmark-service/setup.py benchmark-service/MANIFEST.in  /benchmark-service/

RUN hail-pip-install /benchmark-service && rm -rf /benchmark-service && \
    hail-pip-install plotly pandas scipy numpy

EXPOSE 5000
