ARG DOCKER_PREFIX={{ global.docker_prefix }}
FROM $DOCKER_PREFIX/ubuntu:noble-20251001

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

COPY controller.sh /
COPY curlrc /root/.curlrc
COPY hail-apt-get-install /bin/hail-apt-get-install
COPY hail-pip-install /bin/hail-pip-install
COPY pip.conf /root/.config/pip/pip.conf
COPY retry /bin/retry
COPY uv.toml /root/.config/uv/uv.toml
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
        g++ \
        gcc \
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
