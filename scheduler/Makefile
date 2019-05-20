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
