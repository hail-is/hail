from typer.testing import CliRunner

from hailtop.hailctl.__main__ import app

runner = CliRunner()


def test_emr_is_registered():
    res = runner.invoke(app, ['emr', '--help'])
    assert res.exit_code == 0
    assert 'Amazon EMR' in res.stdout
