export SPARK_CLASSPATH=${PWD}'/build/libs/hail-all-spark.jar'
export PYTHONPATH=${PWD}/python:${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-${PY4J_VERSION}-src.zip
export HAIL_RELEASE=${HAIL_VERSION}-$(git rev-parse --short=12 HEAD)

cd build/tmp/python/hail/docs
make $@
