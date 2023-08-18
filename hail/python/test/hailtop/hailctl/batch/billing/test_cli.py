import pytest
import yaml

from typer.testing import CliRunner

from hailtop.batch_client.client import BatchClient
from hailtop.hailctl.batch.billing import cli


fake_billing_project_1 = {
    'accrued_cost': 0.0,
    'billing_project': 'test',
    'limit': None,
    'status': 'open',
    'users': ['test', 'jdoe'],
}

fake_billing_project_2 = {
    'accrued_cost': 100,
    'billing_project': 'foo',
    'limit': 10000,
    'status': 'open',
    'users': [],
}


@pytest.fixture(autouse=True)
def batch_client(monkeypatch):
    mock_method_results = {
        'get_billing_project': fake_billing_project_1,
        'list_billing_projects': [fake_billing_project_1, fake_billing_project_2],
    }

    for method, result in mock_method_results.items():
        monkeypatch.setattr(BatchClient, method, lambda *_: result)


def test_get_billing_project(runner: CliRunner):
    res = runner.invoke(cli.app, ['get', 'test'], catch_exceptions=False)
    assert res.stdout.strip() == yaml.dump([fake_billing_project_1]).strip()


def test_list_billing_projects(runner: CliRunner):
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.stdout.strip() == yaml.dump([fake_billing_project_1, fake_billing_project_2]).strip()
