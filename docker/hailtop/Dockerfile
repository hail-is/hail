FROM {{ global.docker_prefix }}/python:3.7-slim

COPY docker/hail-ubuntu/retry \
     docker/hail-ubuntu/hail-pip-install \
     /bin

COPY hail/python/hailtop/requirements.txt requirements.txt
RUN hail-pip-install -r requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/
COPY /hail_version /hailtop/hailtop/hail_version
RUN hail-pip-install /hailtop
