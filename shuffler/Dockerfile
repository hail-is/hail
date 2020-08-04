FROM {{ service_base_image.image }}

RUN apt-get update && \
  apt-get -y install \
    openssl \
  && rm -rf /var/lib/apt/lists/* && \
# https://github.com/openssl/openssl/issues/7754#issuecomment-444063355 \
  sed -i 's/^RANDFILE/#RANDFILE/' /etc/ssl/openssl.cnf

COPY hail.jar /

EXPOSE 443
ENTRYPOINT ["java"]
CMD ["-cp", "/spark-2.4.0-bin-hadoop2.7/jars/*:/hail.jar", "is.hail.services.shuffler.server.ShuffleServer"]
