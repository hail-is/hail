.PHONY: test-local test-local-in-cluster \
  clean build


PY_FILES = $(shell find pipeline -iname \*.py -not -exec git check-ignore -q {} \; -print)
PY_CHECKERS = build/flake8 build/pylint

clean:
	-rm -r build

build/conda-env: environment.yml
	conda env update -f environment.yml || conda env create -f environment.yml
	mkdir -p build && touch $@

build/flake8: $(PY_FILES) build/conda-env
	. activate hail-pipeline && python -m flake8 pipeline
	mkdir -p build && touch $@

build/pylint: $(PY_FILES) build/conda-env
	. activate hail-pipeline && python -m pylint --rcfile pipeline/pylintrc pipeline --score=n
	mkdir -p build && touch $@

test-local: build/conda-env $(PY_CHECKERS)
	POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 ./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: build/conda-env $(PY_CHECKERS)
	POD_NAMESPACE='test' ./test-locally.sh
