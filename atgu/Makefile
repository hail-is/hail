EXTRA_PYTHONPATH := ../hail/python:../gear
PYTHON := PYTHONPATH=$${PYTHONPATH:+$${PYTHONPATH}:}$(EXTRA_PYTHONPATH) python3

.PHONY: check
check:
	$(PYTHON) -m flake8  --config ../setup.cfg atgu
	$(PYTHON) -m pylint --rcfile ../pylintrc atgu --score=n
	curlylint .
	bash ../check-sql.sh
