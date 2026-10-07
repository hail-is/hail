import json
from typing import Any, Dict, Optional

import pytest_asyncio

from batch.front_end.query.job_list import JobListLimits, parse_job_list_params
from batch.front_end.query.job_list_sql import get_job_list

from .db_seed import Attempt, Job, JobGroup, Update, analyze_tables, preempted_job, seed_batch

LIMITS = JobListLimits()

ALL_INCLUDES = 'start_time,end_time,latest_attempt_duration,exit_code,attempts.cost_per_hour,parent_ids,cost,total_jobs'


async def job_list(db, batch_id: int, limits: JobListLimits = LIMITS, **query: Any) -> Dict[str, Any]:
    q = {
        k: (json.dumps(v) if k == 'filter' else str(v).lower() if isinstance(v, bool) else str(v))
        for k, v in query.items()
    }
    return await get_job_list(db, batch_id, parse_job_list_params(q, limits), limits)


def ids(resp) -> list:
    return [j['job_id'] for j in resp['jobs']]


async def test_smoke_every_include(db):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[
                    Job(attributes={'shard': '1'}),
                    Job(state='Failed', job_group_id=1),
                    Job(state='Running', job_group_id=2, parent_ids=[1]),
                    preempted_job(state='Ready'),
                    Job(state='Error', name=None),
                    Job(
                        attempts=[
                            Attempt('a1', start_time=1_000, end_time=2_000, reason='preempted'),
                            Attempt('a2', start_time=3_000, end_time=7_000),
                        ]
                    ),
                ],
                job_groups=[JobGroup(), JobGroup(parent_id=1)],
            ),
            Update(jobs=[Job(state='Pending')], committed=False, n_reserved_jobs=2),
        ],
    )
    resp = await job_list(db, seeded.batch_id, job_group_ids=0, recursive=True, include=ALL_INCLUDES)
    assert ids(resp) == seeded.committed_job_ids == [1, 2, 3, 4, 5, 6]
    assert resp['include'] == [
        'start_time',
        'end_time',
        'latest_attempt_duration',
        'exit_code',
        'attempts',
        'attempts.cost_per_hour',
        'parent_ids',
        'cost',
        'total_jobs',
    ]
    pagination = resp['pagination']
    assert pagination['scan_range'] == {'min': 1, 'max': 6, 'stable_below_job_id': 7}
    assert pagination['page_end_reason'] == 'boundary'
    assert pagination['next_page'] is None and pagination['previous_page'] is None
    assert pagination['total_jobs'] == 6

    by_id = {j['job_id']: j for j in resp['jobs']}
    assert by_id[1]['name'] == 'job-1' and by_id[1]['exit_code'] == 0 and by_id[1]['cost'] > 0
    assert by_id[2]['exit_code'] == 1 and by_id[2]['job_group_id'] == 1
    assert by_id[3]['parent_ids'] == [1] and by_id[3]['end_time'] is None
    assert by_id[3]['latest_attempt_duration'] is None
    assert by_id[4]['state'] == 'Ready' and by_id[4]['latest_attempt_duration'] is None
    assert by_id[4]['start_time'] is not None and len(by_id[4]['attempts']) == 1
    assert by_id[5]['name'] is None and by_id[5]['exit_code'] is None
    six = by_id[6]
    assert six['start_time'] == 1_000 and six['end_time'] == 7_000 and six['latest_attempt_duration'] == 4_000
    assert [a['attempt_id'] for a in six['attempts']] == ['a1', 'a2']
    assert all(a['cost_per_hour'] is not None and a['cost_per_hour'] > 0 for a in six['attempts'])
    for j in resp['jobs']:
        assert j['parent_ids_truncated'] is False

    nothing: Optional[Dict[str, Any]] = await job_list(db, seeded.batch_id)
    assert nothing is not None
    assert nothing['include'] == []
    for j in nothing['jobs']:
        for k in ('start_time', 'end_time', 'exit_code', 'attempts', 'parent_ids', 'parent_ids_truncated', 'cost'):
            assert j[k] is None, k


# Scoping: every statement shape reads one batch, on any plan, and keeps its time limit


LARGE_N_JOBS = 20_000


@pytest_asyncio.fixture(scope='session')
async def large_batch(db):
    """A batch big enough that MySQL plans as it would in production.

    On a few hundred rows MySQL reads the whole batch through another index and sorts it, since that's cheap there;
    plan choice depends on data size, so the scoping checks run here. No costs: their trigger dominates seeding.
    """
    states = ['Success', 'Failed', 'Error', 'Cancelled', 'Success', 'Success', 'Success', 'Running']
    jobs = [
        Job(
            state=states[i % len(states)],
            job_group_id=(i // 1000) % 4,
            attributes={'shard': str(i % 10)},
            parent_ids=[1] if i % 50 == 1 else [],
        )
        for i in range(LARGE_N_JOBS)
    ]
    seeded = await seed_batch(
        db,
        [
            Update(jobs=jobs, job_groups=[JobGroup(), JobGroup(parent_id=1), JobGroup()]),
            Update(jobs=[Job(state='Pending', job_group_id=2) for _ in range(20)], committed=False, n_reserved_jobs=30),
        ],
        with_costs=False,
    )
    await analyze_tables(db)
    return seeded


def _scoping_statements(batch_id: int):
    from batch.front_end.query.job_list import parse_filter, parse_include  # pylint: disable=import-outside-toplevel
    from batch.front_end.query.job_list_sql import (  # pylint: disable=import-outside-toplevel
        GroupFilter,
        attempts_statement,
        batch_range_statement,
        cost_per_hour_statement,
        count_statement,
        direct_group_range_statement,
        groups_statement,
        jobs_statement,
        parent_edges_statement,
        recursive_group_range_statement,
        staged_pending_jobs_statement,
        subgroups_statement,
    )

    def leaf(field, op, value=None, **kw):
        d = {'field': field, 'op': op, **kw}
        if op != 'exists':
            d['value'] = value
        return parse_filter(json.dumps(d), LIMITS)

    filters = {
        'none': None,
        'state': leaf('state', '=', 'Failed'),
        'name': leaf('name', '=', 'job-3'),
        'attribute': leaf('attribute', '=', '1', key='shard'),
        'attribute_exists': leaf('attribute', 'exists', key='shard'),
        'text': leaf('text', 'contains', 'shard'),
        'text_exact': leaf('text', '=', 'job-3'),
        'instance': leaf('instance', '=', 'seed-standard'),
        'instance_collection': leaf('instance_collection', '=', 'standard'),
        'exit_code': leaf('exit_code', 'in', [0, 1]),
        'cost': leaf('cost', '>', 0),
        'end_time': leaf('end_time', '>', 0),
        'duration': leaf('duration', '>', 0),
        'or_fields': parse_filter(
            json.dumps({
                'or': [{'field': 'state', 'op': '=', 'value': 'Failed'}, {'field': 'cost', 'op': '>', 'value': 1}]
            }),
            LIMITS,
        ),
    }
    table_include = parse_include('start_time,end_time,latest_attempt_duration,exit_code,cost')
    windows = {'whole': (1, LARGE_N_JOBS), 'mid': (LARGE_N_JOBS // 2, LARGE_N_JOBS // 2 + 4999)}
    groups = {
        'root': None,
        'direct': GroupFilter((1,), False),
        'recursive': GroupFilter((1, 2), True),
    }
    stmts = {
        'groups': groups_statement(batch_id, [0, 1]),
        'batch_range': batch_range_statement(batch_id),
        'direct_group_range': direct_group_range_statement(batch_id, [1, 2]),
        'staged_pending_jobs': staged_pending_jobs_statement(batch_id, [1, 2]),
        'subgroups': subgroups_statement(batch_id, [1], 10001),
        'recursive_group_range': recursive_group_range_statement(batch_id, [1]),
        'attempts': attempts_statement(batch_id, [1, 2, 3]),
        'cost_per_hour': cost_per_hour_statement(batch_id, [1, 2, 3]),
        'parent_edges_forward': parent_edges_statement(batch_id, [5, 6, 7], 'forward', 10001),
        'parent_edges_backward': parent_edges_statement(batch_id, [7, 6, 5], 'backward', 10001),
    }
    for fname, filter_ in filters.items():
        for gname, group_filter in groups.items():
            for direction in ('forward', 'backward'):
                for window_name, (lo, hi) in windows.items():
                    stmts[f'jobs_{fname}_{gname}_{direction}_{window_name}'] = jobs_statement(
                        batch_id, lo, hi, direction, 51, group_filter, filter_, table_include
                    )
            stmts[f'count_{fname}_{gname}'] = count_statement(batch_id, 1, 10_000, group_filter, filter_)
    return stmts


async def test_every_statement_scoped(db, noise_batch, large_batch):  # pylint: disable=unused-argument
    from .db_query_checks import assert_hint_kept, assert_scoped, explain  # pylint: disable=import-outside-toplevel

    failures = []
    for name, stmt in _scoping_statements(large_batch.batch_id).items():
        plan = await explain(db, stmt.sql(10_000), stmt.args)
        # For a direct group MySQL may intersect the group index with the primary key, which loses job id
        # order. That's bounded by the window, and the row-read tests bound what it actually reads.
        allow_filesort = name.startswith('jobs_') and '_direct_' in name
        try:
            assert_scoped(plan, aliases=stmt.aliases, allow_filesort=allow_filesort)
            assert_hint_kept(plan)
        except AssertionError as e:
            failures.append(f'--- {name}\n{str(e)[:3000]}')
    assert not failures, '\n'.join(failures)
