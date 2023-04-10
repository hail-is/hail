FROM --platform=linux/amd64 {{ global.docker_root_image }}

ENV LANG C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
  apt-get -y install \
    git \
    htop \
    unzip bzip2 zip tar \
    wget curl \
    rsync \
    emacs-nox \
    mysql-client \
    xsltproc pandoc \
    jq \
    python \
    python3.8 python3-pip python3.8-dev \
    liblapack3 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
  rm -rf /var/lib/apt/lists/*

RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add - && \
  echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list && \
  apt update && \
  apt -y install temurin-11-jdk

COPY requirements.txt .
RUN mkdir hailtop
COPY hailtop_requirements.txt hailtop/requirements.txt
RUN python3 -m pip install --upgrade --no-cache-dir --upgrade pip && \
  python3 -m pip install --upgrade --no-cache-dir setuptools && \
  python3 -m pip install --upgrade --no-cache-dir -r requirements.txt && \
  python3 -m pip install --upgrade --no-cache-dir aiomysql && \
  python3 -m pip check

ARG HAIL_WHEEL
COPY $HAIL_WHEEL .
RUN python3 -m pip install --upgrade --no-cache-dir --quiet $HAIL_WHEEL && \
  python3 -m pip check

ARG BENCHMARK_WHEEL
COPY $BENCHMARK_WHEEL .
RUN python3 -m pip install --upgrade --no-cache-dir --quiet $BENCHMARK_WHEEL && \
  python3 -m pip check
