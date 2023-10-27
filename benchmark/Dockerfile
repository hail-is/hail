ARG HAIL_UBUNTU
FROM --platform=linux/amd64 ${HAIL_UBUNTU}

RUN hail-apt-get-install \
    liblapack3 \
    openjdk-11-jre-headless \
    wget

RUN --mount=src=benchmark/python/pinned-requirements.txt,target=benchmark-requirements.txt \
    --mount=src=hail/python/pinned-requirements.txt,target=hail-requirements.txt \
    hail-pip-install -r benchmark-requirements.txt -r hail-requirements.txt

ARG HAIL_BENCHMARK_VERSION
ARG HAIL_WHEEL
ENV HAIL_BENCHMARK_VERSION=${HAIL_BENCHMARK_VERSION}
RUN --mount=src=${HAIL_WHEEL},target=${HAIL_WHEEL} \
    --mount=src=benchmark/python,target=/benchmark,rw \
    hail-pip-install --no-deps ${HAIL_WHEEL} /benchmark
