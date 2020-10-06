FROM {{ service_base_image.image }}

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/
RUN hail-pip-install /batch && rm -rf /batch

EXPOSE 5000
