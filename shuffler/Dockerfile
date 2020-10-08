FROM {{ service_base_image.image }}

COPY hail.jar /

EXPOSE 443
CMD ["/bin/bash" "-c" "exec java -cp ${SPARK_HOME}/jars/*:/hail.jar is.hail.services.shuffler.server.ShuffleServer"]
