FROM {{ service_base_image.image }}

COPY atgu/setup.py atgu/MANIFEST.in /atgu/
COPY atgu/atgu /atgu/atgu/
RUN pip3 install --no-cache-dir /atgu && \
  rm -rf /atgu

EXPOSE 5000
