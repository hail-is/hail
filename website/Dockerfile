ARG BASE_IMAGE={{ hail_ubuntu_image.image }}
FROM $BASE_IMAGE

COPY hail/python/hailtop/pinned-requirements.txt hailtop-requirements.txt
COPY gear/pinned-requirements.txt gear-requirements.txt
COPY web_common/pinned-requirements.txt web_common-requirements.txt
RUN hail-pip-install \
      -r hailtop-requirements.txt \
      -r gear-requirements.txt \
      -r web_common-requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/

COPY gear/setup.py /gear/setup.py
COPY gear/gear /gear/gear/

COPY web_common/setup.py web_common/MANIFEST.in /web_common/
COPY web_common/web_common /web_common/web_common/

COPY website/MANIFEST.in website/setup.py /website/
COPY /website/website /website/website

COPY docs.tar.gz /
RUN cd /website/website && \
    tar -xvzf /docs.tar.gz --no-same-owner && \
    hail-pip-install /hailtop /gear /web_common /website && \
    chmod -R 0444 $(python3 -m pip show website | grep -E '^Location: ' | sed 's/Location: //')

CMD ["python3", "-m", "website"]
