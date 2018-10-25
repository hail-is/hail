FROM docker:dind
MAINTAINER hail-team@broadinstitute.org

RUN apk --no-cache add \
    bash \
    curl \
    python

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

ADD images fetch-image.sh ./

CMD /bin/sh fetch-image.sh
