FROM {{ hail_ubuntu_image.image }}

COPY hail/python/hailtop/pinned-requirements.txt requirements.txt
RUN hail-pip-install -r requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/
COPY /hail_version /hailtop/hailtop/hail_version
RUN hail-pip-install /hailtop
