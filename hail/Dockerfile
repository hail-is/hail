FROM ubuntu:18.04

RUN apt-get update && \
  apt-get -y install \
    openjdk-8-jdk-headless \
    python3 \
    python3-pip \
  && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install ipython
COPY *.whl .
RUN python3 -m pip --no-cache-dir install *.whl

ENTRYPOINT ["ipython"]
CMD []
