FROM gcr.io/{{ global.project }}/ubuntu:focal-20201106
ENV LANG C.UTF-8
COPY retry /bin/retry
COPY hail-apt-get-install /bin/hail-apt-get-install
# We need software-properties-common for add-apt-repository; however, that
# package depends on Python 3.8, which installs PyGObject with missing
# dependencies. This fails pip check, so we remove it after we're done.
RUN echo "APT::Acquire::Retries \"5\";" > /etc/apt/apt.conf.d/80-retries && \
    hail-apt-get-install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove -y software-properties-common
RUN hail-apt-get-install python3.7-minimal python3.7-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    python3 -m pip check && \
    python3 -m pip --version
COPY pip.conf /root/.config/pip/pip.conf
COPY hail-pip-install /bin/hail-pip-install
COPY controller.sh /
