FROM {{ service_base_image.image }}

COPY ci/setup.py ci/MANIFEST.in /ci/
COPY ci/ci /ci/ci/
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /ci && \
  rm -rf /ci && \
  python3 -m pip check

EXPOSE 5000

CMD ["python3", "-m", "ci"]
