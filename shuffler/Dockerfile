FROM {{ service_base_image.image }}

RUN apt-get update && \
  apt-get -y install \
    openssl \
  && rm -rf /var/lib/apt/lists/* && \
  sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf

COPY start-shuffler.sh /
COPY hail.jar /

EXPOSE 5000
