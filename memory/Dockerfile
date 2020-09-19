FROM {{ service_base_image.image }}

COPY memory/setup.py /memory/
COPY memory/memory /memory/memory/
RUN pip3 install --no-cache-dir /memory && \
  rm -rf /memory

EXPOSE 5000
