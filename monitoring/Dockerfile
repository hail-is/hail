FROM {{ hail_ubuntu_image.image }}

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
COPY /hail_version /hailtop/hailtop/hail_version

COPY gear/setup.py /gear/setup.py
COPY gear/gear /gear/gear/

COPY web_common/setup.py web_common/MANIFEST.in /web_common/
COPY web_common/web_common /web_common/web_common/

COPY monitoring/setup.py monitoring/MANIFEST.in  /monitoring/
COPY monitoring/monitoring /monitoring/monitoring/

RUN hail-pip-install /hailtop /gear /web_common /monitoring

EXPOSE 5000
