FROM alpine:3.8

RUN apk --no-cache add \
    python2 \
    py2-pip \
    python3 \
    py3-pip \
    bash \
    git \
    openssh \
    curl \
    make \
    && \
    pip2 --no-cache-dir install --upgrade twine wheel \
    && \
    pip3 --no-cache-dir install --upgrade twine wheel

# this seems easier than getting the keys right for apt
#
# source: https://cloud.google.com/storage/docs/gsutil_install#linux
RUN /bin/sh -c 'curl https://sdk.cloud.google.com | bash' && \
    /root/google-cloud-sdk/bin/gcloud components install beta
ENV PATH $PATH:/root/google-cloud-sdk/bin

VOLUME /secrets
WORKDIR /
