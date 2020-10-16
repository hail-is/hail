FROM ubuntu:bionic-20200921
ENV LANG C.UTF-8
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3-pip && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    python3 -m pip install --upgrade --no-cache-dir pip
COPY pip.conf /root/.config/pip/pip.conf
COPY hail-pip-install /bin/hail-pip-install
