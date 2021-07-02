FROM {{ service_base_image.image }}

# re: RANDFILE, https://github.com/openssl/openssl/issues/7754#issuecomment-444063355
RUN hail-apt-get-install openssl && \
    sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf

COPY config.yaml .
COPY create_certs.py .
