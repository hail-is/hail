HAIL_PYTHON3 ?= python3
PARALLELISM ?= 2

.PHONY: test
test:
	sbt compile
	PYSPARK_SUBMIT_ARGS='--driver-memory 8g --conf spark.kryoserializer.buffer.max=1g pyspark-shell' \
	  PARALLELISM=$(PARALLELISM) \
	  HAIL_PYTHON3=$(HAIL_PYTHON3) \
	  ./test-shuffler.sh

.PHONY: check
check:
	$(HAIL_PYTHON3) -m flake8 shuffler
	$(HAIL_PYTHON3) -m pylint shuffler --rcfile ../pylintrc --score=n

.PHONY: jar
jar:
	sbt assembly

.PHONY: image
image: jar
	docker build -f Dockerfile target/scala-2.11 -t gcr.io/hail-vdc/shuffler

.PHONY: push
push: image
	docker push gcr.io/hail-vdc/shuffler

.PHONY: clean
clean:
	rm -rf build
