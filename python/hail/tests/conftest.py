import pytest

@pytest.fixture(scope="session", autouse=True)
def init(doctest_namespace):
    yield # needed for regular tests to not run the setup for the doctests