import pytest


fails_test = pytest.mark.xfail(True, reason="doesn't yet work", strict=True)
