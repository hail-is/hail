FROM {{ service_base_image.image }}

COPY memory/setup.py /memory/
COPY memory/memory /memory/memory/
RUN hail-pip-install /memory && rm -rf /memory

EXPOSE 5000
