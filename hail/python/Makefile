PYTHON := python3

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
	$(PYTHON) -m mypy --config-file ../../setup.cfg hailtop
