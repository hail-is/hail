build/install: $(shell find hailjwt -iname \*.py)
	python setup.py install
	@mkdir -p $(@D)
	touch $@

.PHONY: test
test: build/install
	python -m pytest ${PYTEST_ARGS:- -v --failed-first} test
