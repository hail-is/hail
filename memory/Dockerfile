FROM {{ hail_ubuntu_image.image }}

COPY hail/python/hailtop/pinned-requirements.txt hailtop-requirements.txt
COPY gear/pinned-requirements.txt gear-requirements.txt
COPY memory/pinned-requirements.txt memory-requirements.txt
RUN hail-pip-install \
      -r hailtop-requirements.txt \
      -r gear-requirements.txt \
      -r memory-requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/
COPY /hail_version /hailtop/hailtop/hail_version

COPY gear/setup.py /gear/setup.py
COPY gear/gear /gear/gear/

COPY memory/setup.py /memory/
COPY memory/memory /memory/memory/

RUN hail-pip-install /hailtop /gear /memory

EXPOSE 5000
