FROM {{ service_base_image.image }}

COPY memory/setup.py /memory/
COPY memory/memory /memory/memory/
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /memory && \
  rm -rf /memory && \
  python3 -m pip check

EXPOSE 5000
