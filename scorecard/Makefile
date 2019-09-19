PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../gear:../web_common:../hail/python python3

PY_FILES := $(shell find scorecard -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 scorecard
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc scorecard --score=n
	touch $@

.PHONY: check
check: flake8-stmp pylint-stmp

.PHONY:
clean:
	rm -f flake8-stmp pylint-stmp
