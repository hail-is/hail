PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}../hail/python:../gear:../web_common
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 notebook
	$(PYTHON) -m pylint --rcfile ../pylintrc notebook --score=n
