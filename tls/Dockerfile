FROM {{ hail_ubuntu_image.image }}

# source: https://cloud.google.com/storage/docs/gsutil_install#linux
# re: RANDFILE, https://github.com/openssl/openssl/issues/7754#issuecomment-444063355
RUN curl -sSLO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-334.0.0-linux-x86_64.tar.gz && \
    tar -xf google-cloud-sdk-334.0.0-linux-x86_64.tar.gz && \
    curl -sSLO https://dl.k8s.io/release/v1.19.7/bin/linux/amd64/kubectl && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    hail-apt-get-install openssl && \
    sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf

ENV PATH $PATH:/google-cloud-sdk/bin

COPY config.yaml .
COPY create_certs.py .
