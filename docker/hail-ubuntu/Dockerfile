FROM debian:11-slim
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
    hail-apt-get-install curl gpg && \
    hail-apt-get-install python3-pip && \
    python3 -m pip install 'pip>=21<22' && \
    python3 -m pip check && \
    python3 -m pip --version
