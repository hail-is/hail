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
