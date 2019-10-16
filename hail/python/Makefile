PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8 --config ../../setup.cfg hailtop
	$(PYTHON) -m pylint --rcfile ../../pylintrc hailtop --score=n
