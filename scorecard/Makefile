EXTRA_PYTHONPATH := ../gear:../web_common:../hail/python
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 scorecard
	$(PYTHON) -m pylint --rcfile ../pylintrc scorecard --score=n
