import http
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import gidgethub

from ci.events import MAX_MESSAGE_LENGTH, CIEvent, record_event, truncate_message
from ci.github import PR, describe_exception
from hailtop.utils import CalledProcessError

SQL_DIR = Path(__file__).parent.parent / 'sql'


def _sql_enum_values(sql: str) -> list:
    m = re.search(r'CREATE TABLE (?:IF NOT EXISTS )?`?ci_events`?.*?event ENUM\((.*?)\) NOT NULL', sql, re.DOTALL)
    assert m is not None
    return re.findall(r"'([a-z_]+)'", m.group(1))


def test_python_enum_matches_migration():
    assert _sql_enum_values((SQL_DIR / '008-ci-events.sql').read_text()) == [e.value for e in CIEvent]


def test_python_enum_matches_estimated_current():
    assert _sql_enum_values((SQL_DIR / 'estimated-current.sql').read_text()) == [e.value for e in CIEvent]


def test_truncate_message_keeps_short_messages():
    assert truncate_message('short') == 'short'


def test_truncate_message_keeps_the_tail():
    message = 'head' + 'x' * MAX_MESSAGE_LENGTH + 'tail'
    truncated = truncate_message(message)
    assert len(truncated) == MAX_MESSAGE_LENGTH
    assert truncated.startswith('[truncated]')
    assert truncated.endswith('tail')


async def test_record_event_swallows_failures():
    db = MagicMock()
    db.execute_insertone = AsyncMock(side_effect=RuntimeError('db down'))
    await record_event(db, CIEvent.CI_STARTED, message='hello')
    db.execute_insertone.assert_awaited_once()


def test_describe_exception_includes_process_output():
    e = CalledProcessError(['git', 'merge'], 1, (b'Auto-merging foo', b'CONFLICT (content): Merge conflict in foo'))
    message = describe_exception(e)
    assert 'status 1' in message
    assert 'Auto-merging foo' in message
    assert 'CONFLICT (content)' in message


def test_describe_exception_other_exceptions():
    assert describe_exception(ValueError('bad build.yaml')) == 'ValueError: bad build.yaml\n'


def _make_pr():
    target_branch = MagicMock()
    target_branch.branch.short_str.return_value = 'hail-is/hail:main'
    target_branch.sha = 'cafef00d'
    with patch('ci.github.TRACKED_PRS'):
        return PR(
            number=123,
            title='Test PR',
            body='',
            source_branch=MagicMock(),
            source_sha='deadbeef',
            target_branch=target_branch,
            author='testuser',
            assignees=set(),
            reviewers=set(),
            labels=set(),
            developers=[],
        )


async def test_request_build_records_reason_changes_only():
    pr = _make_pr()
    with patch('ci.github.record_event', new_callable=AsyncMock) as record:
        await pr.request_build(MagicMock(), 'new commit deadbeef')
        await pr.request_build(MagicMock(), 'new commit deadbeef')
        await pr.request_build(MagicMock(), 'retry by someone')
    assert pr.pending_build_reason == 'retry by someone'
    assert [c.args[1] for c in record.await_args_list] == [CIEvent.BUILD_REQUESTED, CIEvent.BUILD_REQUESTED]
    assert [c.kwargs['message'] for c in record.await_args_list] == ['new commit deadbeef', 'retry by someone']


async def test_request_build_does_not_record_unknown():
    pr = _make_pr()
    pr.pending_build_reason = 'initial build'
    with patch('ci.github.record_event', new_callable=AsyncMock) as record:
        await pr.request_build(MagicMock(), 'unknown')
    assert pr.pending_build_reason == 'unknown'
    record.assert_not_awaited()


async def test_merge_success_records_requested_then_succeeded():
    pr = _make_pr()
    gh = MagicMock()
    gh.put = AsyncMock()
    with patch('ci.github.record_event', new_callable=AsyncMock) as record:
        assert await pr.merge(MagicMock(), gh)
    assert [c.args[1] for c in record.await_args_list] == [CIEvent.MERGE_REQUESTED, CIEvent.MERGE_SUCCEEDED]
    assert record.await_args_list[0].kwargs['target_sha'] == 'cafef00d'


async def test_merge_failure_records_github_error():
    pr = _make_pr()
    gh = MagicMock()
    gh.put = AsyncMock(side_effect=gidgethub.BadRequest(http.HTTPStatus(405), 'Base branch was modified'))
    with patch('ci.github.record_event', new_callable=AsyncMock) as record:
        assert not await pr.merge(MagicMock(), gh)
    assert [c.args[1] for c in record.await_args_list] == [CIEvent.MERGE_REQUESTED, CIEvent.MERGE_FAILED]
    assert record.await_args_list[1].kwargs['message'] == '405 BadRequest: Base branch was modified'
