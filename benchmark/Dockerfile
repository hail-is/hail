ARG HAIL_UBUNTU
FROM --platform=linux/amd64 ${HAIL_UBUNTU}

RUN hail-apt-get-install \
    liblapack3 \
    openjdk-11-jre-headless \
    wget

COPY benchmark/python/pinned-requirements.txt benchmark-requirements.txt
COPY hail/python/pinned-requirements.txt hail-requirements.txt
RUN hail-pip-install \
    -r benchmark-requirements.txt \
    -r hail-requirements.txt

ARG BENCHMARK_WHEEL
COPY ${BENCHMARK_WHEEL} .

ARG HAIL_WHEEL
COPY ${HAIL_WHEEL} .

RUN hail-pip-install --no-deps --quiet \
    $(basename ${BENCHMARK_WHEEL}) \
    $(basename ${HAIL_WHEEL})
