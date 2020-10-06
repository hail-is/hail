FROM {{ service_base_image.image }}

COPY address/setup.py address/MANIFEST.in /address/
COPY address/address /address/address/
RUN pip3 install --use-feature=2020-resolver --upgrade --no-cache-dir /address && \
  rm -rf /address && \
  python3 -m pip check

EXPOSE 5000
