PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH)../gear:../web_common:../hail/python python3

.PHONY: check
check:
	$(PYTHON) -m flake8 scorecard
	$(PYTHON) -m pylint --rcfile ../pylintrc scorecard --score=n
