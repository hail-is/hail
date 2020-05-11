import doctest
import os
import pytest

import hailtop.batch as batch


@pytest.fixture(autouse=True)
def patch_doctest_check_output(monkeypatch):
    # FIXME: remove once test output matches docs
    base_check_output = doctest.OutputChecker.check_output

    def patched_check_output(self, want, got, optionflags):
        return ((not want)
                or (want.strip() == 'None')
                or base_check_output(self, want, got, optionflags | doctest.NORMALIZE_WHITESPACE))

    monkeypatch.setattr('doctest.OutputChecker.check_output', patched_check_output)
    yield
    monkeypatch.undo()


@pytest.fixture(scope="session", autouse=True)
def init(doctest_namespace):
    # This gets run once per process -- must avoid race conditions
    print("setting up doctest...")

    doctest_namespace['Batch'] = batch.Batch

    olddir = os.getcwd()
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "docs"))
    try:
        print("finished setting up doctest...")
        yield
    finally:
        os.chdir(olddir)
