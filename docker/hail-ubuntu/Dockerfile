FROM ubuntu:18.04
ENV LANG C.UTF-8
RUN python3 -m pip install --upgrade --no-cache-dir pip
COPY pip.conf /root/.config/pip/pip.conf
COPY hail-pip-install /bin/hail-pip-install
