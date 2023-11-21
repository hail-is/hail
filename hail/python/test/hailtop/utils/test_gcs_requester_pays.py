import pytest

from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.fs.router_fs import RouterFS
from hailtop.utils import async_to_blocking
from hailtop.utils.gcs_requester_pays import GCSRequesterPaysFSCache


@pytest.mark.parametrize("cls", [RouterFS, RouterAsyncFS])
def test_get_fs_by_requester_pays_config(cls):
    config_1 = "foo"
    config_2 = ("foo", ["bar", "baz", "bat"])
    kwargs_1 = {"gcs_requester_pays_configuration": config_1}
    kwargs_2 = {"gcs_requester_pays_configuration": config_2}
    fses = GCSRequesterPaysFSCache(cls)
    assert fses[None]._gcs_kwargs == {}
    assert fses[config_1]._gcs_kwargs == kwargs_1
    set_kwargs = fses[config_2]._gcs_kwargs
    assert set_kwargs["gcs_requester_pays_configuration"][0] == config_2[0]
    assert len(set_kwargs["gcs_requester_pays_configuration"][1]) == len(config_2[1])
    assert set(set_kwargs["gcs_requester_pays_configuration"][1]) == set(config_2[1])
    default_kwargs_fses = GCSRequesterPaysFSCache(cls, {"gcs_kwargs": kwargs_1})
    assert default_kwargs_fses[None]._gcs_kwargs == kwargs_1
