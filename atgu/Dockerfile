FROM {{ service_base_image.image }}

COPY atgu/setup.py atgu/MANIFEST.in /atgu/
COPY atgu/atgu /atgu/atgu/
RUN hail-pip-install /atgu && rm -rf /atgu

EXPOSE 5000
