EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 web_common
	$(PYTHON) -m pylint --rcfile ../pylintrc web_common --score=n
	curlylint .
