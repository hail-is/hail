PROJECT = $(shell gcloud config get-value project)
DOMAIN ?= hail.is

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../hail/python python3

PY_FILES := $(shell find web_common -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 web_common
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc web_common --score=n
	touch $@

.PHONY: check
check: flake8-stmp pylint-stmp

.PHONY:
clean:
	rm -f flake8-stmp pylint-stmp
