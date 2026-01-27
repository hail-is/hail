ARG DOCKER_PREFIX={{ global.docker_prefix }}
ARG SKIP_GCLOUD_PROFILER=false
FROM $DOCKER_PREFIX/ubuntu:noble-20260113 AS initializer

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

COPY docker/hail-ubuntu/controller.sh /
COPY docker/hail-ubuntu/curlrc /root/.curlrc
COPY docker/hail-ubuntu/hail-apt-get-install /bin/hail-apt-get-install
COPY docker/hail-ubuntu/hail-pip-install /bin/hail-pip-install
COPY docker/hail-ubuntu/pip.conf /root/.config/pip/pip.conf
COPY docker/hail-ubuntu/retry /bin/retry
COPY docker/hail-ubuntu/uv.toml /root/.config/uv/uv.toml
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ARG PYTHON_VERSION=3.11
RUN chmod 755 /bin/retry && \
    chmod 755 /bin/hail-apt-get-install && \
    chmod 755 /bin/hail-pip-install && \
    chmod 755 /controller.sh && \
    echo "APT::Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries && \
    mkdir -p /usr/share/keyrings/ && \
    hail-apt-get-install \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        jq \
        rsync \
        && \
    curl 'https://packages.cloud.google.com/apt/doc/apt-key.gpg' \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' \
        >> /etc/apt/sources.list

ENV UV_PYTHON=$PYTHON_VERSION UV_MANAGED_PYTHON=1 UV_PYTHON_INSTALL_DIR=/opt/uv/python VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
RUN uv venv --seed $VIRTUAL_ENV && \
    command -v python3 && \
    python3 -m pip check && \
    python3 -m pip --version

FROM initializer AS wheel-builder
ARG SKIP_GCLOUD_PROFILER=false

RUN mkdir -p /tmp/wheels
RUN hail-apt-get-install g++ gcc
# Only install google-cloud-profiler if SKIP_GCLOUD_PROFILER is not set to true
RUN --mount=type=bind,source=gear/pinned-requirements.txt,target=/tmp/gear-requirements.txt,readonly \
    if [ "$SKIP_GCLOUD_PROFILER" != "true" ]; then \
        hail-pip-install google-cloud-profiler --constraint /tmp/gear-requirements.txt && \
        pip wheel --wheel-dir=/tmp/wheels google-cloud-profiler; \
    else \
        touch /tmp/wheels/.no-profiler; \
    fi

FROM initializer
ARG SKIP_GCLOUD_PROFILER=false
# Copy google-cloud-profiler wheel if it exists
COPY --from=wheel-builder /tmp/wheels/ /tmp/wheels/
# Only install google-cloud-profiler if the wheel exists and SKIP_GCLOUD_PROFILER is not true
RUN --mount=type=bind,source=gear/pinned-requirements.txt,target=/tmp/gear-requirements.txt,readonly \
    if [ "$SKIP_GCLOUD_PROFILER" != "true" ] && [ -f /tmp/wheels/google_cloud_profiler*.whl ]; then \
        hail-pip-install /tmp/wheels/google_cloud_profiler*.whl --constraint /tmp/gear-requirements.txt; \
    fi
