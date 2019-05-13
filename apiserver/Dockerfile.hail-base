FROM {{ hail_run_image.image }}

COPY hail.zip .
RUN unzip -q hail.zip && \
  mv hail inner && \
  mkdir -p /hail/python && \
  mv inner /hail/python/hail

COPY hail.jar /hail/jars/hail.jar
RUN cp /hail/jars/hail.jar $SPARK_HOME/jars/hail.jar

ENV HAIL_HOME /hail
ENV PYTHONPATH "${PYTHONPATH:+${PYTHONPATH}:}$HAIL_HOME/python"
ENV PYSPARK_SUBMIT_ARGS "--conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator pyspark-shell"
