FROM gcr.io/{{ global.project }}/ubuntu:bionic-20200921
ENV LANG C.UTF-8
COPY hail-apt-get-install /bin/hail-apt-get-install
RUN echo "APT::Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries && \
    hail-apt-get-install python3.7 python3.7-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    python3 -m pip install --upgrade --no-cache-dir pip
COPY pip.conf /root/.config/pip/pip.conf
COPY hail-pip-install /bin/hail-pip-install
