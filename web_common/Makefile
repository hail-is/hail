EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3
BLACK := $(PYTHON) -m black . --line-length=120 --skip-string-normalization

.PHONY: check
check:
	$(PYTHON) -m flake8 web_common
	$(PYTHON) -m pylint --rcfile ../pylintrc web_common --score=n
	$(BLACK) --check --diff
	curlylint .
