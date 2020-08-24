FROM {{ service_base_image.image }}

COPY hail.jar /

EXPOSE 443
ENTRYPOINT ["java"]
CMD ["-cp", "/spark-2.4.0-bin-hadoop2.7/jars/*:/hail.jar", "is.hail.services.shuffler.server.ShuffleServer"]
