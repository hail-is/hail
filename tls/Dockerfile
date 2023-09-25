ARG BASE_IMAGE={{ hail_ubuntu_image.image }}
FROM $BASE_IMAGE

# re: RANDFILE, https://github.com/openssl/openssl/issues/7754#issuecomment-444063355
# jdk not strictly necessary, but we want keytool
RUN curl -LO https://dl.k8s.io/release/v1.21.14/bin/linux/amd64/kubectl && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    hail-apt-get-install openssl openjdk-8-jdk-headless && \
    sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf && \
    hail-pip-install pyyaml

COPY config.yaml /
COPY create_certs.py create_test_db_config.sh /
