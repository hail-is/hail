.PHONY: check

PY_FILES := $(shell find apiserver -iname \*.py -not -exec git check-ignore -q {} \; -print)

PYTHONPATH := $${PYTHONPATH:+:$$PYTHONPATH}../hail/python:../gear:$(shell ls $$SPARK_HOME/python/lib/py4j-*-src.zip):$SPARK_HOME/python
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 apiserver
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc apiserver --score=n
	touch $@

check: flake8-stmp pylint-stmp
