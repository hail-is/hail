PYTHONPATH := $${PYTHONPATH:+$${PYTHONPATH}:}
PYTHON := PYTHONPATH=$(PYTHONPATH) python3

.PHONY: check-hail
check-hail:
	$(PYTHON) -m flake8 --config ../../setup.cfg hail
	$(PYTHON) -m pylint --rcfile ../../pylintrc hail --score=n

# pylint on hail is still a work in progress
.PHONY: check
check:
	$(PYTHON) -m flake8 --config ../../setup.cfg hail
	$(PYTHON) -m flake8 --config ../../setup.cfg hailtop
	$(PYTHON) -m pylint --rcfile ../../pylintrc hailtop --score=n
	cd hailtop && $(PYTHON) -m mypy batch google_storage.py
