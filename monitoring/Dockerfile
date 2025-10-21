ARG BASE_IMAGE={{ hail_ubuntu_image.image }}

# Build wheels using g++ / gcc that we don't want in the final image:
FROM $BASE_IMAGE as wheel-builder
RUN mkdir -p /tmp/wheels
RUN hail-apt-get-install g++ gcc
RUN hail-pip-install google-cloud-profiler
RUN pip wheel --wheel-dir=/tmp/wheels google-cloud-profiler

# Install the wheels and build the final image:
FROM $BASE_IMAGE
COPY --from=wheel-builder /tmp/wheels/google_cloud_profiler*.whl /tmp/wheels/
RUN hail-pip-install /tmp/wheels/google_cloud_profiler*.whl

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

COPY gear/pyproject.toml /gear/pyproject.toml
COPY gear/gear /gear/gear/

COPY web_common/pyproject.toml web_common/MANIFEST.in /web_common/
COPY web_common/web_common /web_common/web_common/

COPY monitoring/setup.py monitoring/MANIFEST.in  /monitoring/
COPY monitoring/monitoring /monitoring/monitoring/

RUN hail-pip-install /hailtop /gear /web_common /monitoring

EXPOSE 5000
