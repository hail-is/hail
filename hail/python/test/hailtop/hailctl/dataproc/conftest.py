from unittest.mock import Mock

import pytest

from hailtop.hailctl.dataproc.cli import MINIMUM_REQUIRED_GCLOUD_VERSION


@pytest.fixture
def gcloud_config():
    """Fixture for gcloud configuration values."""
    return {
        "account": "test@hail.is",
        "project": "hailctl-dataproc-tests",
        "dataproc/region": "us-central1",
        "compute/zone": "us-central1-b",
    }


@pytest.fixture
def gcloud_run():
    return Mock()


@pytest.fixture
def gcloud_output():
    return Mock(return_value="")


@pytest.fixture
def machine_type_info():
    """Fixture for (vCPUs, memory GiB) of machine types used in tests."""
    return {
        "n4-standard-8": (8, 32),
        "n4-standard-16": (16, 64),
        "n4-standard-32": (32, 128),
        "n4-highmem-8": (8, 64),
        "n4-highmem-16": (16, 128),
    }


@pytest.fixture(autouse=True)
def patch_gcloud(monkeypatch, gcloud_run, gcloud_output, gcloud_config, machine_type_info):
    """Automatically replace gcloud functions with mocks."""
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.run", gcloud_run)
    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.output", gcloud_output)
    monkeypatch.setattr(
        "hailtop.hailctl.dataproc.gcloud.get_version", Mock(return_value=MINIMUM_REQUIRED_GCLOUD_VERSION)
    )

    def mock_gcloud_get_config(setting):
        return gcloud_config.get(setting, None)

    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_config", mock_gcloud_get_config)

    def mock_get_machine_type_info(machine_type):
        return machine_type_info[machine_type]

    monkeypatch.setattr("hailtop.hailctl.dataproc.gcloud.get_machine_type_info", mock_get_machine_type_info)

    yield

    monkeypatch.undo()


@pytest.fixture
def deploy_metadata():
    """Fixture for deploy.yaml values."""
    return {
        "wheel": "gs://hail-common/hailctl/dataproc/test-version/hail-test-version-py3-none-any.whl",
        "init_notebook.py": "gs://hail-common/hailctl/dataproc/test-version/init_notebook.py",
        "pip_dependencies": "aiohttp>=3.6,<3.7|aiohttp_session>=2.7,<2.8|asyncinit>=0.2.4,<0.3|bokeh>1.1,<1.3|decorator<5|humanize==1.0.0|hurry.filesize==0.9|nest_asyncio|numpy<2|pandas>0.24,<0.26|parsimonious<0.9|PyJWT|python-json-logger==0.1.11|requests>=2.21.0,<2.21.1|scipy>1.2,<1.4|tabulate==0.8.9|tqdm==4.42.1|jproperties==2.1.1",
        "vep-GRCh37.sh": "gs://hail-common/hailctl/dataproc/test-version/vep-GRCh37.sh",
        "vep-GRCh38.sh": "gs://hail-common/hailctl/dataproc/test-version/vep-GRCh38.sh",
    }
