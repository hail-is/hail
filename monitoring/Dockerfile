FROM {{ service_base_image.image }}

COPY monitoring/monitoring /monitoring/monitoring/

COPY monitoring/setup.py monitoring/MANIFEST.in  /monitoring/

RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /monitoring && \
    rm -rf /monitoring && \
  python3 -m pip check

EXPOSE 5000
