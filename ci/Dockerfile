FROM {{ service_base_image.image }}

COPY ci/setup.py ci/MANIFEST.in /ci/
COPY ci/ci /ci/ci/
RUN hail-pip-isntall /ci && rm -rf /ci

EXPOSE 5000

CMD ["python3", "-m", "ci"]
