ARG BASE_IMAGE={{ hail_ubuntu_image.image }}
FROM $BASE_IMAGE

RUN hail-apt-get-install git

COPY hail/python/hailtop/pinned-requirements.txt hailtop-requirements.txt
COPY gear/pinned-requirements.txt gear-requirements.txt
COPY web_common/pinned-requirements.txt web_common-requirements.txt
COPY ci/pinned-requirements.txt ci-requirements.txt
RUN hail-pip-install \
      -r hailtop-requirements.txt \
      -r gear-requirements.txt \
      -r web_common-requirements.txt \
      -r ci-requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/

COPY gear/setup.py /gear/setup.py
COPY gear/gear /gear/gear/

COPY web_common/setup.py web_common/MANIFEST.in /web_common/
COPY web_common/web_common /web_common/web_common/

COPY ci/setup.py ci/MANIFEST.in /ci/
COPY ci/ci /ci/ci/

RUN hail-pip-install /hailtop /gear /web_common /ci

EXPOSE 5000
