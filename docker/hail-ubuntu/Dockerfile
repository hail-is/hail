ARG DOCKER_PREFIX={{ global.docker_prefix }}
FROM $DOCKER_PREFIX/ubuntu:noble-20250529

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

COPY retry /bin/retry
COPY hail-apt-get-install /bin/hail-apt-get-install
COPY pip.conf /root/.config/pip/pip.conf
COPY hail-pip-install /bin/hail-pip-install
COPY controller.sh /
COPY curlrc /root/.curlrc

ARG PYTHON_VERSION=3.11
# save PYTHON_VERSION to the environment so that dependent images can inherit the value
ENV HAIL_PYTHON_MAJOR_VERSION=$PYTHON_VERSION
RUN chmod 755 /bin/retry && \
    chmod 755 /bin/hail-apt-get-install && \
    chmod 755 /bin/hail-pip-install && \
    chmod 755 /controller.sh && \
    echo "APT::Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries && \
    mkdir -p /usr/share/keyrings/ && \
    hail-apt-get-install curl apt-transport-https ca-certificates gnupg jq rsync && \
    curl 'https://keyserver.ubuntu.com/pks/lookup?search=0xF23C5A6CF475977595C89F51BA6932366A755776&hash=on&exact=on&options=mr&op=get' \
         | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/deadsnakes-ppa-archive-keyring.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu noble main' \
         >> /etc/apt/sources.list && \
    echo 'deb-src [signed-by=/usr/share/keyrings/deadsnakes-ppa-archive-keyring.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu noble main' \
         >> /etc/apt/sources.list && \
    curl 'https://packages.cloud.google.com/apt/doc/apt-key.gpg' \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' \
        >> /etc/apt/sources.list && \
    hail-apt-get-install python$PYTHON_VERSION-minimal python$PYTHON_VERSION-dev python$PYTHON_VERSION-distutils gcc g++ && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_VERSION 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install 'pip>=23,<26' && \
    python3 -m pip check && \
    python3 -m pip --version
