.PHONY: test-local test-local-in-cluster clean build

IGNORED_FILE_PATTERN := \( -name '*~' -o -name '*.log' \)
BATCH_PY_FILES = $(shell find ../batch -iname \*.py -not $(IGNORED_FILE_PATTERN) -print)
PIPELINE_PY_FILES = $(shell find pipeline -iname \*.py -not -exec git check-ignore -q {} \; -print)
PY_CHECKERS = build/flake8 build/pylint

clean:
	-rm -r build

build/conda-env: environment.yml
	conda env update -f environment.yml || conda env create -f environment.yml
	mkdir -p build && touch $@

build/batch: $(BATCH_PY_FILES) build/conda-env
	. ../loadconda && conda activate hail-pipeline && pip install -U ../batch/
	mkdir -p build && touch $@

build/flake8: $(PIPELINE_PY_FILES) build/conda-env build/batch
	. ../loadconda && conda activate hail-pipeline && python -m flake8 pipeline
	mkdir -p build && touch $@

build/pylint: $(PIPELINE_PY_FILES) build/conda-env build/batch
	. ../loadconda && conda activate hail-pipeline && python -m pylint --rcfile pipeline/pylintrc pipeline --score=n
	mkdir -p build && touch $@

test-local: build/conda-env build/batch $(PY_CHECKERS)
	POD_NAMESPACE='test' BATCH_USE_KUBE_CONFIG=1 ./test-locally.sh

# local means server and test client are two processes on one machine
# in-cluster means in a k8s pod (from which we get k8s creds)
test-local-in-cluster: build/conda-env build/batch $(PY_CHECKERS)
	POD_NAMESPACE='test' ./test-locally.sh
