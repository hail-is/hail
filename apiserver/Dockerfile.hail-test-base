FROM {{ hail_run_image.image }}

COPY hail.zip .
RUN unzip -q hail.zip && \
  mv hail inner && \
  mkdir -p /hail/python && \
  mv inner /hail/python/hail

COPY hail-test.jar /hail/jars/hail-test.jar
RUN cp /hail/jars/hail-test.jar $SPARK_HOME/jars/hail-test.jar

ENV HAIL_HOME /hail
ENV PYTHONPATH "${PYTHONPATH:+${PYTHONPATH}:}$HAIL_HOME/python"
ENV PYSPARK_SUBMIT_ARGS "--conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator pyspark-shell"
