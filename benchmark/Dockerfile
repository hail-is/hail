# syntax=docker/dockerfile:1.3.0-labs
# ^ necessary to use an ARG in a --mount value with our current version of buildkit

ARG HAIL_UBUNTU
FROM --platform=linux/amd64 ${HAIL_UBUNTU}

RUN hail-apt-get-install \
    liblapack3 \
    openjdk-11-jre-headless \
    wget

RUN --mount=src=benchmark/python/pinned-requirements.txt,target=benchmark-requirements.txt \
    --mount=src=hail/python/pinned-requirements.txt,target=hail-requirements.txt \
    hail-pip-install -r benchmark-requirements.txt -r hail-requirements.txt

ARG HAIL_WHEEL_DIR
RUN --mount=src=${HAIL_WHEEL_DIR},target=${HAIL_WHEEL_DIR} \
    --mount=src=benchmark/python,target=/benchmark,rw \
    export HAIL_BENCHMARK_VERSION=$(cat benchmark/version) && \
    hail-pip-install --no-deps ${HAIL_WHEEL_DIR}/hail-*-py3-none-any.whl /benchmark
