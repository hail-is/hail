FROM {{ service_base_image.image }}

RUN apt-get update && \
  apt-get -y install \
    liblz4-dev

COPY shuffler-assembly-0.1.jar /shuffler-assembly-0.1.jar

ENTRYPOINT ["java", "-jar", "shuffler-assembly-0.1.jar"]
CMD []
