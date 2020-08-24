FROM {{ service_base_image.image }}

# re: RANDFILE, https://github.com/openssl/openssl/issues/7754#issuecomment-444063355
RUN apt-get update && \
  apt-get -y install \
    openssl \
  && rm -rf /var/lib/apt/lists/* && \
  sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf

COPY config.yaml .
COPY create_certs.py .
