FROM base
MAINTAINER Hail Team <hail@broadinstitute.org>

RUN apt-get update && \
  apt-get -y install \
    git && \
  rm -rf /var/lib/apt/lists/*

RUN mkdir /home/hail-ci && \
    groupadd hail-ci && \
    useradd -g hail-ci hail-ci && \
    chown hail-ci:hail-ci /home/hail-ci

WORKDIR /batch
COPY batch/batch batch
COPY batch/setup.py .

WORKDIR /hail-ci

COPY ci/index.html ci/pr-build-script ci/pr-deploy-script ci/deploy-index.html ./
COPY ci/ci ./ci
COPY ci/run_ci.py ./
RUN chown -R hail-ci:hail-ci ./
RUN pip3 install /batch

USER hail-ci
EXPOSE 5000
VOLUME /hail-ci/oauth-token
VOLUME /hail-ci/gcloud-token
ENTRYPOINT ["python3", "run_ci.py"]
