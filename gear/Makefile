PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 gear
	$(PYTHON) -m pylint --rcfile ../pylintrc gear --score=n
