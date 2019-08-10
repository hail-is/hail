FROM {{ hail_base_image.image }}

COPY apiserver/apiserver /apiserver

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

ENV HAIL_SPARK_PROPERTIES "spark.driver.host=apiserver,spark.driver.bindAddress=0.0.0.0,spark.driver.port=9001,spark.blockManager.port=9002"

CMD ["python3", "/apiserver/apiserver.py"]
