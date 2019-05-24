PYTHON := python3

PY_FILES := $(shell find hailjwt -iname \*.py -not -exec git check-ignore -q {} \; -print)

flake8-stmp: $(PY_FILES)
	python3 -m flake8 hailjwt
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc hailjwt --score=n
	touch $@

check: flake8-stmp pylint-stmp

.PHONY: test
test:
	python3 -m pytest ${PYTEST_ARGS:- -v --failed-first} test

clean:
	rm -f flake8-stmp pylint-stmp
