from typer.testing import CliRunner
from hailtop.hailctl.auth import cli

runner = CliRunner(mix_stderr=False)

def test_print_access_token():
    res = runner.invoke(cli.app, 'print-access-token', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout != ''
