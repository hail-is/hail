import os
import pytest


fails_in_azure = pytest.mark.xfail(
    os.environ.get('HAIL_CLOUD') == 'azure', reason="doesn't yet work on azure", strict=True
)


skip_in_azure = pytest.mark.skipif(os.environ.get('HAIL_CLOUD') == 'azure', reason="not applicable to azure")

run_if_azure = pytest.mark.skipif(os.environ.get('HAIL_CLOUD') != 'azure', reason="only applicable to azure")
