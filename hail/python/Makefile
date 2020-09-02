PYTHON := python3

.PHONY: fleep
fleep:
	$(PYTHON) -m flake8 --config ../../setup.cfg hailtop/aiotools
	$(PYTHON) -m flake8 --config ../../setup.cfg hailtop/aiogoogle
	$(PYTHON) -m pylint --rcfile ../../pylintrc hailtop.aiotools --score=n
	$(PYTHON) -m pylint --rcfile ../../pylintrc hailtop.aiogoogle --score=n

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
	$(PYTHON) -m mypy --config-file ../../setup.cfg hailtop/utils hailtop/batch hailtop/google_storage.py
