FROM {{ service_base_image.image }}

COPY monitoring/monitoring /monitoring/monitoring/

COPY monitoring/setup.py monitoring/MANIFEST.in  /monitoring/

RUN hail-pip-install /monitoring && rm -rf /monitoring

EXPOSE 5000
