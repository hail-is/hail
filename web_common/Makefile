PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../hail/python:../gear python3

.PHONY: check
check:
	$(PYTHON) -m flake8 web_common
	$(PYTHON) -m pylint --rcfile ../pylintrc web_common --score=n
