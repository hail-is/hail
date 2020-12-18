FROM {{ service_base_image.image }}

COPY setup.py MANIFEST.in /batch/
COPY batch /batch/batch/
RUN hail-pip-install /batch && rm -rf /batch
EXPOSE 5000
