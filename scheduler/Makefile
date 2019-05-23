.PHONY: check clean

PYTHON := python3

PY_FILES := $(shell find scheduler -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	python3 -m flake8 scheduler
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc scheduler --score=n
	touch $@

check: flake8-stmp pylint-stmp

clean:
	rm -f flake8-stmp pylint-stmp


# FIXME move this code to jar
.PHONY: all executor submit

SPARK_CP := $(SPARK_HOME)/jars/*
SCHED_CP := .
SUBMIT_CP := $(SPARK_CP):$(SCHED_CP)

all:
	CLASSPATH='$(SPARK_CP)' scalac foo.scala

executor:
	CLASSPATH='$(SCHED_CP)' scala org.apache.spark.foo.Client localhost 2

submit:
	CLASSPATH='$(SUBMIT_CP)' scala org.apache.spark.foo.Submit localhost 1000
