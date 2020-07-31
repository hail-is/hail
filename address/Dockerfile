FROM {{ service_base_image.image }}

COPY address/setup.py address/MANIFEST.in /address/
COPY address/address /address/address/
RUN pip3 install --no-cache-dir /address && \
  rm -rf /address

EXPOSE 5000
