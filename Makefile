# This Makefile just indirects to call gradlew

SPARK_VERSION :=2.2.0
SPARK_HOME    :=/usr/local/java/spark-${SPARK_VERSION}-bin-hadoop2.7

CXX  := /usr/bin/g++ -std=c++14 -Wno-deprecated
FLEX := /usr/bin/flex
LIBFLEX := -ll

.PHONY: shadowJar go test

all: shadowJar go

shadowJar:
	./gradlew -Dspark.version=${SPARK_VERSION} shadowJar archiveZip

nativeLibDebug:
	./gradlew -Dspark.version=${SPARK_VERSION} nativeLibDebug

nativeLibClean:
	./gradlew -Dspark.version=${SPARK_VERSION} nativeLibClean

go:
	./gradlew -Dspark.version=${SPARK_VERSION} -Dspark.home=${SPARK_HOME} \
	  -Dtest.single=AnnotationsSuite test

test:
	./gradlew -Dspark.version=${SPARK_VERSION} -Dspark.home=${SPARK_HOME} test

testPython:
	./gradlew -Dspark.version=${SPARK_VERSION} -Dspark.home=${SPARK_HOME} \
	  -Dtest.parallelism=4 testPython

clean:
	./gradlew clean

scan: scan.lex
	$(FLEX) scan.lex
	$(CXX) -o $@ lex.yy.c ${LIBFLEX}
	
pull_master:
	git checkout master
	git pull https://github.com/hail-is/hail.git master

