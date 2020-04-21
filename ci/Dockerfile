FROM {{ service_base_image.image }}

COPY ci/setup.py ci/MANIFEST.in /ci/
COPY ci/ci /ci/ci/
RUN python3 -m hailpip --no-cache-dir /ci && \
  rm -rf /ci

EXPOSE 5000

CMD ["python3", "-m", "ci"]
