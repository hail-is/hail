from typing import List, Optional
from unittest.mock import patch

from typer.testing import CliRunner

from hailtop.hailctl.batch import cli

runner = CliRunner()


def invoke_change_quote(args: List[str], input: Optional[str] = None):
    with patch('hailtop.batch_client.client.BatchClient') as MockClient:
        client = MockClient.return_value.__enter__.return_value
        client.get_billing_project.return_value = {'quote_name': 'OLD', 'accrued_cost': 1234.5}
        client.change_billing_project_quote.return_value = {'billing_project': 'bp', 'quote_name': 'NEW'}
        result = runner.invoke(cli.app, ['billing', 'projects', 'change-quote', 'bp', 'NEW', *args], input=input)
    return result, client


def test_change_quote_warns_and_moves_when_confirmed():
    result, client = invoke_change_quote([], input='y\n')
    assert result.exit_code == 0, result.output
    assert 'moves its whole spend history ($1,234.50 so far)' in result.output
    client.change_billing_project_quote.assert_called_once_with(
        'bp', 'NEW', acknowledge_history_move=True, comment=None
    )


def test_change_quote_aborts_when_declined():
    result, client = invoke_change_quote([], input='n\n')
    assert result.exit_code != 0
    client.change_billing_project_quote.assert_not_called()


def test_change_quote_aborts_without_input():
    # Non-interactive callers that don't pass --accept-history-move must not move anything.
    result, client = invoke_change_quote([])
    assert result.exit_code != 0
    client.change_billing_project_quote.assert_not_called()


def test_change_quote_flag_skips_prompt():
    result, client = invoke_change_quote(['--accept-history-move'])
    assert result.exit_code == 0, result.output
    client.get_billing_project.assert_not_called()
    client.change_billing_project_quote.assert_called_once_with(
        'bp', 'NEW', acknowledge_history_move=True, comment=None
    )
