FROM {{ service_base_image.image }}

COPY address/setup.py address/MANIFEST.in /address/
COPY address/address /address/address/
RUN hail-pip-install /address && rm -rf /address

EXPOSE 5000
