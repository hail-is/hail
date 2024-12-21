ARG BASE_IMAGE={{ hail_ubuntu_image.image }}
FROM $BASE_IMAGE

# re: RANDFILE, https://github.com/openssl/openssl/issues/7754#issuecomment-444063355
# jdk not strictly necessary, but we want keytool
RUN curl -LO https://dl.k8s.io/release/v1.21.14/bin/linux/amd64/kubectl && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    hail-apt-get-install openssl openjdk-11-jdk-headless && \
    sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf

COPY hail/python/hailtop/pinned-requirements.txt hailtop-requirements.txt
RUN hail-pip-install -r hailtop-requirements.txt

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/MANIFEST.in /hailtop/MANIFEST.in
COPY hail/python/hailtop /hailtop/hailtop/
RUN hail-pip-install /hailtop

COPY tls/config.yaml /
COPY tls/create_certs.py tls/create_test_db_config.sh /
