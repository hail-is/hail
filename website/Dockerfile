FROM {{ service_base_image.image }}

COPY /website /website/website
COPY MANIFEST.in setup.py /website/
COPY docs.tar.gz /
RUN cd /website/website && \
    tar -xvzf /docs.tar.gz --no-same-owner && \
    hail-pip-install /website && \
    chmod -R 0444 $(python3 -m pip show website | grep -E '^Location: ' | sed 's/Location: //')

CMD ["python3", "-m", "website"]
