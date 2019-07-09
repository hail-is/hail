PY_FILES = $(shell find notebook -iname \*.py -not -exec git check-ignore -q {} \; -print)

PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

flake8-stmp: $(PY_FILES)
	$(PYTHON) -m flake8 notebook
	touch $@

pylint-stmp: $(PY_FILES)
	$(PYTHON) -m pylint --rcfile ../pylintrc notebook --score=n
	touch $@

.PHONY: check
check: flake8-stmp pylint-stmp

.PHONY:  clean
clean:
	rm -f flake8-stmp pylint-stmp
