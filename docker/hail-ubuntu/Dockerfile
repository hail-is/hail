FROM {{ global.docker_prefix }}/ubuntu:jammy-20230624
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
COPY retry /bin/retry
COPY hail-apt-get-install /bin/hail-apt-get-install
COPY pip.conf /root/.config/pip/pip.conf
COPY hail-pip-install /bin/hail-pip-install
COPY controller.sh /
COPY curlrc /root/.curlrc
RUN chmod 755 /bin/retry && \
    chmod 755 /bin/hail-apt-get-install && \
    chmod 755 /bin/hail-pip-install && \
    chmod 755 /controller.sh && \
    echo "APT::Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries && \
    mkdir -p /usr/share/keyrings/ && \
    hail-apt-get-install curl gpg jq rsync && \
    curl 'https://keyserver.ubuntu.com/pks/lookup?search=0xF23C5A6CF475977595C89F51BA6932366A755776&hash=on&exact=on&options=mr&op=get' \
         | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/deadsnakes-ppa-archive-keyring.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main' \
         >> /etc/apt/sources.list && \
    echo 'deb-src [signed-by=/usr/share/keyrings/deadsnakes-ppa-archive-keyring.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main' \
         >> /etc/apt/sources.list && \
    hail-apt-get-install python3.9-minimal python3.9-dev python3.9-distutils gcc g++ && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    python3 -m pip install 'pip>=21<22' && \
    python3 -m pip check && \
    python3 -m pip --version
