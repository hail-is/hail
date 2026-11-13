import json
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio

from batch.front_end.query.job_list import JobListLimits, parse_job_list_params
from batch.front_end.query.job_list_sql import get_job_list

from .db_seed import Attempt, Job, JobGroup, Update, analyze_tables, preempted_job, seed_batch

LIMITS = JobListLimits()

ALL_INCLUDES = 'start_time,end_time,latest_attempt_duration,exit_code,attempts.cost_per_hour,parent_ids,cost,total_jobs'


def _query_value(k: str, v: Any) -> str:
    if k == 'filter':
        return json.dumps(v)
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (list, tuple)):
        return ','.join(str(x) for x in v)
    return str(v)


async def job_list(db, batch_id: int, limits: JobListLimits = LIMITS, **query: Any) -> Dict[str, Any]:
    q = {k: _query_value(k, v) for k, v in query.items() if v is not None}
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


# The property test: random filters and paging, against the endpoint's own unfiltered fields


async def _property_batch(db):
    """Nested groups, an abandoned update leaving a gap, a pending update at the end, retries, parents."""
    names = ['train-a', 'train-b', 'eval_x', 'eval%y', 'merge', None]
    terminal = ['Success', 'Failed', 'Error', 'Cancelled', 'Success']

    def job(i: int, first_id: int, group_ids) -> Job:
        state = [*terminal, 'Running', 'Ready'][i % 7]
        kwargs: Dict[str, Any] = {}
        if i % 11 == 3 and state == 'Success':
            kwargs['attempts'] = [
                Attempt(f'p{i}', start_time=1_000 + i, end_time=2_000 + i, reason='preempted', instance_name='spot-1'),
                Attempt(f'q{i}', start_time=3_000 + i, end_time=9_000 + 7 * i, instance_name='worker-2'),
            ]
        if state == 'Failed' and i % 3 == 1:
            kwargs['exit_code'] = None
        if state in ('Ready', 'Running'):
            # earlier Success jobs of this update; some jobs get many, to exercise the parent-edge budget
            successes = [first_id + p for p in range(i) if p % 7 in (0, 4)]
            kwargs['parent_ids'] = successes[: 9 if i % 4 == 0 else 2]
        return Job(
            state=state,
            job_group_id=group_ids[i % len(group_ids)],
            name=names[i % len(names)],
            attributes={'shard': str(i % 5), 'kind': 'gpu' if i % 3 == 0 else 'cpu'},
            **kwargs,
        )

    return await seed_batch(
        db,
        [
            Update(
                jobs=[job(i, 1, [0, 1, 2, 3, 4]) for i in range(80)],
                job_groups=[JobGroup(), JobGroup(parent_id=1), JobGroup(), JobGroup(parent_id=3)],
            ),
            # abandoned: reserves 81-100, uploads ten jobs
            Update(jobs=[Job(state='Pending', job_group_id=2) for _ in range(10)], committed=False, n_reserved_jobs=20),
            Update(jobs=[job(i, 101, [2, 4, 1]) for i in range(70)]),
            # pending at the end
            Update(jobs=[Job(state='Pending', job_group_id=4) for _ in range(5)], committed=False, n_reserved_jobs=8),
        ],
    )


_COMPARE = {
    '<': lambda x, v: x < v,
    '<=': lambda x, v: x <= v,
    '>': lambda x, v: x > v,
    '>=': lambda x, v: x >= v,
}


def _match_value(x: Any, op: str, v: Any) -> bool:
    """One value against a leaf, with SQL's rules: null never matches. Strings compare case-insensitively, as
    MySQL's default collation does; the test data is lower case."""
    if x is None:
        return False
    if isinstance(x, str):
        x = x.lower()
        v = [s.lower() for s in v] if isinstance(v, list) else v.lower() if isinstance(v, str) else v
    if op == '=':
        return x == v
    if op == '!=':
        return x != v
    if op == 'in':
        return x in v
    if op == 'contains':
        return v in x
    if op == 'not_contains':
        return v not in x
    return _COMPARE[op](x, v)


class _Truth:
    """Every committed job's fields as the endpoint reports them, plus the seeded attributes and groups."""

    def __init__(self, seeded, resp):
        self.seeded = seeded
        self.jobs = {j['job_id']: j for j in resp['jobs']}
        self.attributes = {}
        for job_id, j in self.jobs.items():
            attrs = dict(seeded.jobs[job_id].attributes)
            if j['name'] is not None:
                attrs['name'] = j['name']
            self.attributes[job_id] = attrs

    def matches(self, node: Optional[Dict[str, Any]], job_id: int) -> bool:
        if node is None:
            return True
        if 'and' in node:
            return all(self.matches(c, job_id) for c in node['and'])
        if 'or' in node:
            return any(self.matches(c, job_id) for c in node['or'])
        j, attrs = self.jobs[job_id], self.attributes[job_id]
        field, op, v = node['field'], node['op'], node.get('value')
        instances = [a['instance_name'] for a in j['attempts']]
        if field == 'attribute':
            if op == 'exists':
                return node['key'] in attrs
            return _match_value(attrs.get(node['key']), op, v)
        if field == 'text':
            return any(_match_value(s, op, v) for s in [*attrs.keys(), *attrs.values(), *instances])
        if field == 'instance':
            return any(_match_value(s, op, v) for s in instances)
        if field == 'instance_collection':
            return _match_value(self.seeded.jobs[job_id].inst_coll, op, v)
        if field == 'duration':
            x = None if j['end_time'] is None or j['start_time'] is None else j['end_time'] - j['start_time']
            return _match_value(x, op, v)
        return _match_value(j[field], op, v)

    def in_groups(self, job_id: int, job_group_ids, recursive: bool) -> bool:
        g = self.jobs[job_id]['job_group_id']
        groups = self.seeded.ancestors(g) if recursive else [g]
        return any(x in job_group_ids for x in groups)


def _random_leaf(rng, truth: _Truth) -> Dict[str, Any]:
    jobs = list(truth.jobs.values())

    def some(key):
        values = [j[key] for j in jobs if j[key] is not None]
        return rng.choice(values) + rng.choice([-1, 0, 0, 1])

    field = rng.choice([
        'job_id', 'state', 'name', 'attribute', 'text', 'instance', 'instance_collection',
        'exit_code', 'cost', 'start_time', 'end_time', 'duration', 'latest_attempt_duration',
    ])  # fmt: skip
    if field == 'job_id':
        if rng.random() < 0.3:
            return {'field': field, 'op': 'in', 'value': rng.sample(range(1, 190), rng.randint(1, 4))}
        return {'field': field, 'op': rng.choice(['=', '<', '<=', '>', '>=']), 'value': rng.randint(0, 190)}
    if field == 'state':
        states = ['Pending', 'Ready', 'Running', 'Success', 'Failed', 'Error', 'Cancelled']
        if rng.random() < 0.4:
            return {'field': field, 'op': 'in', 'value': rng.sample(states, rng.randint(1, 3))}
        return {'field': field, 'op': rng.choice(['=', '!=']), 'value': rng.choice(states)}
    if field == 'name':
        op = rng.choice(['=', '!=', 'contains', 'not_contains'])
        value = (
            rng.choice(['train-a', 'eval_x', 'merge'])
            if op in ('=', '!=')
            else rng.choice(['train', 'l_x', 'l%y', '_', '%', 'e', 'zz'])
        )
        return {'field': field, 'op': op, 'value': value}
    if field == 'attribute':
        key = rng.choice(['shard', 'kind', 'name', 'missing'])
        op = rng.choice(['=', '!=', 'contains', 'not_contains', 'exists'])
        if op == 'exists':
            return {'field': field, 'key': key, 'op': op}
        return {'field': field, 'key': key, 'op': op, 'value': rng.choice(['0', '3', 'gpu', 'pu', 'train-b', 'x'])}
    if field == 'text':
        op = rng.choice(['contains', '='])
        return {'field': field, 'op': op, 'value': rng.choice(['gpu', 'shard', 'spot-1', 'worker', 'eval_x', '2'])}
    if field == 'instance':
        op = rng.choice(['=', 'contains'])
        return {'field': field, 'op': op, 'value': rng.choice(['spot-1', 'worker-2', 'seed', 'work', 'nope'])}
    if field == 'instance_collection':
        return {'field': field, 'op': '=', 'value': rng.choice(['standard', 'highmem'])}
    if field == 'exit_code':
        if rng.random() < 0.3:
            return {'field': field, 'op': 'in', 'value': rng.sample([0, 1, 3], 2)}
        return {'field': field, 'op': rng.choice(['=', '!=']), 'value': rng.choice([0, 1, 3])}
    if field == 'cost':
        costs = sorted({j['cost'] for j in jobs if j['cost'] is not None})
        # between two costs, so float rounding can't decide it
        i = rng.randrange(len(costs) - 1)
        return {'field': field, 'op': rng.choice(list(_COMPARE)), 'value': (costs[i] + costs[i + 1]) / 2}
    if field == 'duration':
        durations = [j['end_time'] - j['start_time'] for j in jobs if j['end_time'] and j['start_time']]
        return {
            'field': field,
            'op': rng.choice(list(_COMPARE)),
            'value': max(0, rng.choice(durations) + rng.choice([-1, 0, 1])),
        }
    return {'field': field, 'op': rng.choice(list(_COMPARE)), 'value': max(0, some(field))}


def _random_filter(rng, truth: _Truth) -> Optional[Dict[str, Any]]:
    shape = rng.choice(['none', 'leaf', 'and', 'or', 'and_of_ors', 'or_of_ands'])
    if shape == 'none':
        return None
    if shape == 'leaf':
        return _random_leaf(rng, truth)
    if shape in ('and', 'or'):
        return {shape: [_random_leaf(rng, truth) for _ in range(rng.randint(1, 3))]}
    outer, inner = ('and', 'or') if shape == 'and_of_ors' else ('or', 'and')
    return {outer: [{inner: [_random_leaf(rng, truth) for _ in range(rng.randint(1, 2))]} for _ in range(2)]}


PROPERTY_INCLUDE = 'start_time,end_time,latest_attempt_duration,exit_code,cost'


async def _follow_both_ways(db, batch_id: int, limits: JobListLimits, first: Dict[str, Any], truth: _Truth):
    """Every job reached from ``first`` by following links both ways, checking each page on the way."""
    seen = []
    n_requests = 0

    async def page(query):
        nonlocal n_requests
        n_requests += 1
        assert n_requests < 2_000, 'paging did not terminate'
        resp = await job_list(db, batch_id, limits, **query)
        page_ids = ids(resp)
        assert page_ids == sorted(page_ids)
        assert len(page_ids) <= int(first['limit'])
        for j in resp['jobs']:
            expected = truth.jobs[j['job_id']]
            for k in ('job_group_id', 'name', 'state', *PROPERTY_INCLUDE.split(',')):
                assert j[k] == expected[k], (j['job_id'], k, j[k], expected[k])
            if j['parent_ids'] is not None and not j['parent_ids_truncated']:
                assert j['parent_ids'] == expected['parent_ids'], j['job_id']
        seen.extend(page_ids)
        return resp['pagination']

    pagination = await page(first)
    for link_key in ('next_page', 'previous_page'):
        link = pagination[link_key]
        while link is not None:
            p = await page({**first, **link})
            link = p[link_key]
    return seen


@pytest_asyncio.fixture(scope='session')
async def property_truth(db):
    seeded = await _property_batch(db)
    full = await job_list(
        db,
        seeded.batch_id,
        job_group_ids=0,
        recursive=True,
        limit=1000,
        include=f'{PROPERTY_INCLUDE},attempts,parent_ids',
    )
    truth = _Truth(seeded, full)
    assert sorted(truth.jobs) == seeded.committed_job_ids
    return truth


PROPERTY_SEEDS_PER_CASE = 15


@pytest.mark.parametrize('case', range(10))
async def test_property_paging_and_filters(db, property_truth, case):
    import random  # pylint: disable=import-outside-toplevel

    truth = property_truth
    batch_id = truth.seeded.batch_id
    group_choices = [([0], True), ([0], False), ([1], False), ([1], True), ([2], True), ([3, 4], True), ([3, 2], False)]
    for seed in range(case * PROPERTY_SEEDS_PER_CASE, (case + 1) * PROPERTY_SEEDS_PER_CASE):
        rng = random.Random(seed)
        job_group_ids, recursive = rng.choice(group_choices)
        filter_ = _random_filter(rng, truth)
        with_edges = rng.random() < 0.3
        limits = JobListLimits(
            parent_edge_page_max=rng.choice([3, 10]),
            group_range_lookup_max_jobs=rng.choice([5, 20_000]),
            group_range_lookup_max_subgroups=rng.choice([1, 10_000]),
        )
        first = {
            'job_group_ids': job_group_ids,
            'recursive': recursive,
            'filter': filter_,
            'scan_direction': rng.choice(['forward', 'backward']),
            'scan_start_job_id': rng.choice([None, rng.randint(0, 200)]),
            'limit': rng.randint(1, 15),
            'max_scan_size': rng.choice([4, 17, 60, 50_000]),
            'include': PROPERTY_INCLUDE + (',parent_ids' if with_edges else ''),
        }
        seen = await _follow_both_ways(db, batch_id, limits, first, truth)
        expected = sorted(
            job_id
            for job_id in truth.jobs
            if truth.in_groups(job_id, job_group_ids, recursive) and truth.matches(filter_, job_id)
        )
        assert len(seen) == len(set(seen)), (seed, 'a job came back twice')
        assert sorted(seen) == expected, (
            seed,
            first,
            sorted(set(expected) - set(seen)),
            sorted(set(seen) - set(expected)),
        )


# Targeted cases


def _pagination(resp, *keys):
    p = resp['pagination']
    return tuple(p[k] for k in keys)


def _link(start, direction):
    return {'scan_start_job_id': start, 'scan_direction': direction}


async def test_scan_range_gaps_and_stable_below(db):
    seeded = await seed_batch(
        db,
        [
            Update(jobs=[Job(), Job(), Job()]),
            Update(job_groups=[JobGroup()]),  # committed, groups only: no ids
            Update(jobs=[Job(state='Pending'), Job(state='Pending')], committed=False, n_reserved_jobs=5),
            Update(jobs=[Job(), Job()]),
            Update(job_groups=[JobGroup()], committed=False),  # pending, groups only: ignored
        ],
    )
    resp = await job_list(db, seeded.batch_id, job_group_ids=0, recursive=True, include='total_jobs')
    assert ids(resp) == [1, 2, 3, 9, 10]
    # committed ids go above batches.n_jobs (5)
    assert resp['pagination']['scan_range'] == {'min': 1, 'max': 10, 'stable_below_job_id': 4}
    assert resp['pagination']['total_jobs'] == 5

    done = await seed_batch(db, [Update(jobs=[Job(), Job()]), Update(jobs=[Job()])])
    resp = await job_list(db, done.batch_id, job_group_ids=0, recursive=True)
    assert resp['pagination']['scan_range'] == {'min': 1, 'max': 3, 'stable_below_job_id': 4}

    # only a pending update has jobs: nothing to scan, and nothing below 1 can still appear
    waiting = await seed_batch(
        db, [Update(job_groups=[JobGroup()]), Update(jobs=[Job(state='Pending')], committed=False)]
    )
    resp = await job_list(db, waiting.batch_id, job_group_ids=0, recursive=True, include='total_jobs')
    assert resp['jobs'] == []
    assert resp['pagination']['scan_range'] == {'min': None, 'max': None, 'stable_below_job_id': 1}
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page', 'total_jobs') == (
        'boundary',
        None,
        None,
        0,
    )

    empty = await seed_batch(db, [Update(job_groups=[JobGroup()])])
    resp = await job_list(db, empty.batch_id, job_group_ids=0, recursive=True)
    assert resp['pagination']['scan_range'] == {'min': None, 'max': None, 'stable_below_job_id': 1}


async def _contiguous_group_batch(db):
    """Jobs 40-60 are group 1, the rest the root; 100 jobs."""
    return await seed_batch(
        db,
        [Update(jobs=[Job(job_group_id=1 if 40 <= i + 1 <= 60 else 0) for i in range(100)], job_groups=[JobGroup()])],
    )


async def test_windows_anchored_and_outside_the_range(db):
    seeded = await _contiguous_group_batch(db)
    group = {'job_group_ids': 1, 'recursive': False}
    b = seeded.batch_id

    resp = await job_list(db, b, **group)
    assert ids(resp) == list(range(40, 61))
    assert resp['pagination']['scan_range'] == {'min': 40, 'max': 60, 'stable_below_job_id': 101}

    # a block starting below the group keeps its own end
    resp = await job_list(db, b, **group, scan_start_job_id=31, max_scan_size=20)
    assert ids(resp) == list(range(40, 51))
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == (
        'scan_size',
        _link(51, 'forward'),
        None,
    )

    # entirely before the range: jump in
    resp = await job_list(db, b, **group, scan_start_job_id=1, max_scan_size=30)
    assert ids(resp) == []
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == (
        'scan_size',
        _link(40, 'forward'),
        None,
    )
    resp = await job_list(db, b, **group, scan_direction='backward', scan_start_job_id=200, max_scan_size=10)
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == (
        'scan_size',
        None,
        _link(60, 'backward'),
    )

    # entirely past it: a link back in
    resp = await job_list(db, b, **group, scan_start_job_id=61)
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == (
        'boundary',
        None,
        _link(60, 'backward'),
    )
    resp = await job_list(db, b, **group, scan_direction='backward', scan_start_job_id=39)
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == (
        'boundary',
        _link(40, 'forward'),
        None,
    )

    # backward with no start begins at the end; results ascend
    resp = await job_list(db, b, **group, scan_direction='backward', limit=5)
    assert ids(resp) == [56, 57, 58, 59, 60]
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == ('limit', None, _link(55, 'backward'))

    # exactly `limit` rows left before the edge
    resp = await job_list(db, b, **group, scan_start_job_id=56, limit=5)
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == (
        'boundary',
        None,
        _link(55, 'backward'),
    )


async def test_job_id_leaves_narrow_the_range(db):
    seeded = await _contiguous_group_batch(db)
    b = seeded.batch_id
    root = {'job_group_ids': 0, 'recursive': True}

    def job_id(op, v):
        return {'field': 'job_id', 'op': op, 'value': v}

    resp = await job_list(db, b, **root, filter={'and': [job_id('>=', 10), job_id('<', 21)]})
    assert ids(resp) == list(range(10, 21))
    assert resp['pagination']['scan_range'] == {'min': 10, 'max': 20, 'stable_below_job_id': 101}
    assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page') == ('boundary', None, None)

    resp = await job_list(db, b, **root, filter=job_id('in', [70, 30]), include='total_jobs')
    assert ids(resp) == [30, 70]
    assert resp['pagination']['scan_range']['min'] == 30 and resp['pagination']['scan_range']['max'] == 70

    # under an `or`, no narrowing
    resp = await job_list(db, b, **root, filter={'or': [job_id('=', 7), job_id('=', 9)]})
    assert ids(resp) == [7, 9]
    assert resp['pagination']['scan_range']['min'] == 1 and resp['pagination']['scan_range']['max'] == 100

    for narrowed_to_nothing in (job_id('<', 1), {'and': [job_id('>=', 90), job_id('<=', 10)]}, job_id('>', 100)):
        resp = await job_list(db, b, **root, filter=narrowed_to_nothing, scan_start_job_id=50, include='total_jobs')
        assert resp['jobs'] == []
        assert resp['pagination']['scan_range'] == {'min': None, 'max': None, 'stable_below_job_id': 101}
        assert _pagination(resp, 'page_end_reason', 'next_page', 'previous_page', 'total_jobs') == (
            'boundary',
            None,
            None,
            0,
        )


async def test_limit_zero_is_metadata_only(db):
    seeded = await _contiguous_group_batch(db)
    resp = await job_list(db, seeded.batch_id, job_group_ids=1, limit=0, include='total_jobs')
    assert resp['jobs'] == []
    assert _pagination(resp, 'first_job_id', 'last_job_id', 'next_page', 'previous_page', 'page_end_reason') == (
        None,
        None,
        None,
        None,
        None,
    )
    assert resp['pagination']['scan_range'] == {'min': 40, 'max': 60, 'stable_below_job_id': 101}
    assert resp['pagination']['total_jobs'] == 21


async def test_group_ranges_and_fallbacks(db):
    # 1 and 3 are root children; 2 is under 1. Jobs: 1-10 root, 11-20 group 1, 21-30 group 2, 31-40 group 3.
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[Job(job_group_id=g) for g in [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10],
                job_groups=[JobGroup(), JobGroup(parent_id=1), JobGroup(), JobGroup(parent_id=1)],
            ),
            # pending, under group 2
            Update(jobs=[Job(state='Pending', job_group_id=2) for _ in range(6)], committed=False),
        ],
    )
    b = seeded.batch_id

    async def scan_range(limits=LIMITS, **q):
        resp = await job_list(db, b, limits, **q)
        r = resp['pagination']['scan_range']
        return ids(resp), (r['min'], r['max'])

    assert await scan_range(job_group_ids=1, recursive=False) == (list(range(11, 21)), (11, 20))
    assert await scan_range(job_group_ids=1, recursive=True) == (list(range(11, 31)), (11, 46))  # includes pending ids
    assert await scan_range(job_group_ids=[3, 1], recursive=False) == ([*range(11, 21), *range(31, 41)], (11, 40))
    assert await scan_range(job_group_ids=4, recursive=True) == ([], (None, None))  # empty group

    batch_wide = (1, 40)
    # committed jobs over the gate
    small = JobListLimits(group_range_lookup_max_jobs=19)
    assert (await scan_range(small, job_group_ids=1, recursive=True))[1] == batch_wide
    # committed under it, but pending staged jobs push it over
    small = JobListLimits(group_range_lookup_max_jobs=25)
    assert (await scan_range(small, job_group_ids=1, recursive=True))[1] == batch_wide
    assert (await scan_range(small, job_group_ids=3, recursive=True))[1] == (31, 40)
    # too many sub-groups (empty ones count)
    small = JobListLimits(group_range_lookup_max_subgroups=1)
    assert (await scan_range(small, job_group_ids=1, recursive=True))[1] == batch_wide
    assert (await scan_range(small, job_group_ids=3, recursive=True))[1] == (31, 40)
    # overlapping groups: each job once; one group over the gate makes the whole request batch-wide
    small = JobListLimits(group_range_lookup_max_jobs=25)
    assert await scan_range(small, job_group_ids=[1, 2, 3], recursive=True) == (list(range(11, 41)), batch_wide)

    # the root, recursive: batch-wide with no gate, alone or in a list, with no group filter
    tiny = JobListLimits(group_range_lookup_max_jobs=1, group_range_lookup_max_subgroups=1)
    assert await scan_range(tiny, job_group_ids=0, recursive=True) == (list(range(1, 41)), batch_wide)
    assert await scan_range(tiny, job_group_ids=[3, 0], recursive=True) == (list(range(1, 41)), batch_wide)
    assert await scan_range(job_group_ids=0, recursive=False) == (list(range(1, 11)), (1, 10))


async def test_unknown_or_uncommitted_group_is_not_found(db):
    from batch.front_end.query.job_list_sql import JobGroupNotFound  # pylint: disable=import-outside-toplevel

    seeded = await seed_batch(
        db, [Update(jobs=[Job()], job_groups=[JobGroup()]), Update(job_groups=[JobGroup()], committed=False)]
    )
    for groups in (9, [1, 9], 2):
        with pytest.raises(JobGroupNotFound):
            await job_list(db, seeded.batch_id, job_group_ids=groups)
    with pytest.raises(JobGroupNotFound):
        await job_list(db, seeded.batch_id + 1000)


async def test_total_jobs(db):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[
                    Job(state='Failed' if i % 4 == 0 else 'Success', job_group_id=1 if i < 8 else 2 if i < 12 else 0)
                    for i in range(20)
                ],
                job_groups=[JobGroup(), JobGroup()],
            ),
            Update(jobs=[Job(state='Pending', job_group_id=1) for _ in range(30)], committed=False),
        ],
    )
    b = seeded.batch_id
    failed = {'field': 'state', 'op': '=', 'value': 'Failed'}

    async def total(limits=LIMITS, **q):
        return (await job_list(db, b, limits, limit=0, include='total_jobs', **q))['pagination']['total_jobs']

    # exact within a narrow enough range, filtered or not; the pending update isn't counted
    assert await total(job_group_ids=0, recursive=True) == 20
    assert await total(job_group_ids=0, recursive=True, filter=failed) == 5
    assert await total(job_group_ids=1, recursive=True) == 8
    assert await total(job_group_ids=1, recursive=False, filter=failed) == 2

    # too wide to count: the exact counters where they exist
    narrow = JobListLimits(total_jobs_count_max=5)
    assert await total(narrow, job_group_ids=0, recursive=True) == 20
    assert await total(narrow, job_group_ids=1, recursive=True) == 8
    assert await total(narrow, job_group_ids=0, recursive=True, filter=failed) is None
    assert await total(narrow, job_group_ids=1, recursive=False) is None
    # a small group in a wide batch is narrow enough (jobs 9-12)
    assert await total(JobListLimits(total_jobs_count_max=4), job_group_ids=2, recursive=False, filter=failed) == 1
    # a direct group's range includes the pending update's ids (21-50), so it's too wide however few are committed
    assert await total(JobListLimits(total_jobs_count_max=8), job_group_ids=1, recursive=False) is None

    # not requested
    assert (await job_list(db, b, limit=0))['pagination']['total_jobs'] is None


async def test_exit_code_semantics(db):
    from batch.exceptions import QueryError  # pylint: disable=import-outside-toplevel

    seeded = await seed_batch(db, [Update(jobs=[Job(state='Error'), Job(), Job(state='Failed'), Job(state='Running')])])
    b = seeded.batch_id
    resp = await job_list(db, b, include='exit_code')
    assert [j['exit_code'] for j in resp['jobs']] == [None, 0, 1, None]

    async def matching(op, value):
        return ids(await job_list(db, b, filter={'field': 'exit_code', 'op': op, 'value': value}))

    # the Error job's unknown exit code is a JSON null: it matches nothing
    assert await matching('=', 0) == [2]
    assert await matching('!=', 1) == [2]
    assert await matching('in', [0, 1]) == [2, 3]

    v1 = await seed_batch(db, [Update(jobs=[Job(), Job(state='Failed')])], format_version=1)
    assert [j['exit_code'] for j in (await job_list(db, v1.batch_id, include='exit_code'))['jobs']] == [0, 1]
    with pytest.raises(QueryError):
        await job_list(db, v1.batch_id, filter={'field': 'exit_code', 'op': '=', 'value': 0})


async def test_cost_and_time_fields(db):
    seeded = await seed_batch(
        db,
        [
            Update(
                jobs=[
                    Job(state='Ready'),  # never ran
                    preempted_job(state='Ready'),  # waiting to be retried
                    preempted_job(state='Cancelled'),  # preempted, then cancelled while waiting
                    Job(state='Cancelled'),  # cancelled before any attempt
                    Job(
                        attempts=[
                            Attempt('a', start_time=1_000, end_time=2_000, reason='preempted'),
                            Attempt('b', start_time=5_000, end_time=8_000),
                        ]
                    ),
                    Job(state='Running', attempts=[Attempt('c', start_time=4_000)]),
                ]
            )
        ],
    )
    b = seeded.batch_id
    resp = await job_list(db, b, include='start_time,end_time,latest_attempt_duration,cost,attempts')
    fields = {
        j['job_id']: (j['state'], j['start_time'], j['end_time'], j['latest_attempt_duration'], len(j['attempts']))
        for j in resp['jobs']
    }
    assert fields[1] == ('Ready', None, None, None, 0)
    assert fields[2][0] == 'Ready' and fields[2][1] is not None and fields[2][2:] == (None, None, 1)
    assert fields[3][0] == 'Cancelled' and fields[3][1] is not None and fields[3][2:] == (None, None, 1)
    assert fields[4] == ('Cancelled', None, None, None, 0)
    assert fields[5] == ('Success', 1_000, 8_000, 3_000, 2)
    assert fields[6] == ('Running', 4_000, None, None, 1)

    costs = {j['job_id']: j['cost'] for j in resp['jobs']}
    assert costs[1] is None and costs[4] is None and costs[5] > 0

    async def matching(field, op, value):
        return ids(await job_list(db, b, filter={'field': field, 'op': op, 'value': value}))

    assert 1 not in await matching('cost', '<', 1_000_000)
    assert await matching('duration', '>=', 0) == [5]  # wall clock 7,000; the preempted-then-cancelled job has no end
    assert await matching('latest_attempt_duration', '<=', 3_000) == [5]
    assert await matching('end_time', '>', 0) == [5]
    # generated attempts start far later, at seeding's T0
    assert await matching('start_time', '<', 4_001) == [5, 6]
    assert await matching('start_time', '<', 4_000) == [5]


async def test_attempt_rates(db):
    seeded = await seed_batch(db, [Update(jobs=[Job(), Job(state='Ready')])], with_costs=False)
    resp = await job_list(db, seeded.batch_id, include='attempts.cost_per_hour')
    assert resp['include'] == ['attempts', 'attempts.cost_per_hour']
    by_id = {j['job_id']: j for j in resp['jobs']}
    assert [a['cost_per_hour'] for a in by_id[1]['attempts']] == [None]  # no resource rows
    assert by_id[2]['attempts'] == []


async def test_parent_edge_budget(db):
    jobs = [Job() for _ in range(6)]  # 1-6: parents
    jobs += [
        Job(parent_ids=[1, 2, 3, 4, 5]),  # 7: over a budget of 3 on its own
        Job(parent_ids=[1, 2, 3]),  # 8: exactly the budget
        Job(),  # 9: no parents
        Job(parent_ids=[4]),  # 10
        Job(parent_ids=[5, 6]),  # 11
        Job(parent_ids=[1, 2, 3, 4]),  # 12
    ]
    seeded = await seed_batch(db, [Update(jobs=jobs)])
    b = seeded.batch_id
    limits = JobListLimits(parent_edge_page_max=3)

    async def page(**q):
        resp = await job_list(db, b, limits, include='parent_ids', **q)
        return (
            [(j['job_id'], j['parent_ids'], j['parent_ids_truncated']) for j in resp['jobs']],
            resp['pagination']['page_end_reason'],
            resp['pagination']['next_page'],
            resp['pagination']['previous_page'],
        )

    rows, reason, nxt, _ = await page(scan_start_job_id=7)
    assert rows == [(7, [1, 2, 3], True)] and reason == 'parent_edges' and nxt == _link(8, 'forward')
    rows, reason, nxt, _ = await page(scan_start_job_id=8)
    assert rows == [(8, [1, 2, 3], False), (9, [], False)] and reason == 'parent_edges' and nxt == _link(10, 'forward')
    # 10 and 11 fit; 12 is only the lookahead row, so the page ended on its limit
    rows, reason, nxt, _ = await page(scan_start_job_id=10, limit=2)
    assert [r[0] for r in rows] == [10, 11] and reason == 'limit' and nxt == _link(12, 'forward')
    # backward: scan order is highest first, so 12 alone exceeds the budget; edges are read highest first too
    rows, reason, _, prev = await page(scan_start_job_id=12, scan_direction='backward')
    assert rows == [(12, [2, 3, 4], True)] and reason == 'parent_edges' and prev == _link(11, 'backward')
    rows, reason, _, prev = await page(scan_start_job_id=11, scan_direction='backward')
    assert [r[0] for r in rows] == [9, 10, 11] and reason == 'parent_edges' and prev == _link(8, 'backward')


# Row reads: what each statement shape actually reads, whatever plan MySQL picks


async def test_row_reads_bounded(db, noise_batch, large_batch):  # pylint: disable=unused-argument
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

    from .db_query_checks import count_row_reads  # pylint: disable=import-outside-toplevel

    b = large_batch.batch_id
    table = parse_include('start_time,end_time,latest_attempt_duration,exit_code,cost')
    never_text = parse_filter(json.dumps({'field': 'text', 'op': 'contains', 'value': 'zzz'}), LIMITS)
    never_time = parse_filter(json.dumps({'field': 'end_time', 'op': '<', 'value': 1}), LIMITS)
    page_ids = list(range(1, 51))
    # Group 1 holds ids 1001-2000, 5001-6000, ...; group 2, under it, 2001-3000, ...: 10,000 jobs between them.
    cases = [
        # (statement, bound): a dense page stops at its limit, whatever the window
        (jobs_statement(b, 1, LARGE_N_JOBS, 'forward', 51, None, None, table), 15 * 51),
        (jobs_statement(b, 1, LARGE_N_JOBS, 'backward', 51, None, None, table), 15 * 51),
        # a filter matching nothing reads the window, not the batch
        (jobs_statement(b, 8001, 10000, 'forward', 51, None, never_text, table), 8 * 2000),
        (jobs_statement(b, 8001, 10000, 'backward', 51, None, never_time, table), 8 * 2000),
        (jobs_statement(b, 10000, 14999, 'forward', 51, GroupFilter((1,), False), None, table), 1.5 * 5000),
        (jobs_statement(b, 8001, 10000, 'forward', 51, GroupFilter((1,), True), None, table), 2 * 2000),
        (count_statement(b, 8001, 10000, None, never_text), 8 * 2000),
        (batch_range_statement(b), 20),
        (groups_statement(b, [0, 1]), 20),
        (direct_group_range_statement(b, [1, 2]), 20),
        (staged_pending_jobs_statement(b, [1, 2]), 20),
        (subgroups_statement(b, [0], 10_001), 20),
        (recursive_group_range_statement(b, [1]), 1.2 * 10_000 + 50),
        (attempts_statement(b, page_ids), 3 * len(page_ids)),
        (cost_per_hour_statement(b, page_ids), 3 * len(page_ids)),
        (parent_edges_statement(b, page_ids, 'forward', 10_001), 3 * len(page_ids)),
    ]
    over = []
    for stmt, bound in cases:
        _, reads = await count_row_reads(db, stmt.sql(10_000), stmt.args)
        if reads > bound:
            over.append(f'{stmt.name}: {reads} reads > {bound}')
    assert not over, over


# The time budget: one deadline per request, enforced by MySQL itself


class _Clock:
    """Returns ``times`` in order, then the last one forever: the first call starts the budget, then one per
    statement."""

    def __init__(self, *times: float):
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


async def _job_list_with_clock(db, batch_id, clock, limits=LIMITS, **query):
    q = {k: _query_value(k, v) for k, v in query.items() if v is not None}
    return await get_job_list(db, batch_id, parse_job_list_params(q, limits), limits, clock=clock)


NEVER_MATCHES = {'field': 'text', 'op': 'contains', 'value': 'zzz'}


async def test_statement_interrupted_by_mysql(db, large_batch):
    from batch.front_end.query.job_list_sql import MYSQL_QUERY_TIMEOUT, JobListTimeout  # pylint: disable=C0415

    # groups and range with the full 10 s; then 2 ms for the jobs query, which scans 20,000 ids
    clock = _Clock(0, 0, 0, 9.998)
    with pytest.raises(JobListTimeout) as e:
        await _job_list_with_clock(
            db, large_batch.batch_id, clock, job_group_ids=0, recursive=True, filter=NEVER_MATCHES
        )
    # MySQL interrupted it, rather than the budget check stopping it
    assert e.value.__cause__ is not None and e.value.__cause__.args[0] == MYSQL_QUERY_TIMEOUT


async def test_budget_spent_before_the_next_statement(db, large_batch):
    from batch.front_end.query.job_list_sql import JobListTimeout  # pylint: disable=import-outside-toplevel

    clock = _Clock(0, 0, 0, 10.5)
    with pytest.raises(JobListTimeout) as e:
        await _job_list_with_clock(db, large_batch.batch_id, clock, job_group_ids=0, recursive=True)
    assert e.value.__cause__ is None


# The count always covers the whole scan range, whatever the page's window; either way the page is returned.
@pytest.mark.parametrize('max_scan_size, page_end_reason', [(5_000, 'scan_size'), (None, 'boundary')])
async def test_count_gets_only_the_leftover_budget(db, large_batch, max_scan_size, page_end_reason):
    wide = JobListLimits(total_jobs_count_max=LARGE_N_JOBS)
    query = {
        'job_group_ids': 0,
        'recursive': True,
        'filter': NEVER_MATCHES,
        'include': 'total_jobs',
        'limit': 5,
        'max_scan_size': max_scan_size,
    }

    # the page gets the budget it needs; the count, scanning the whole batch, is left 2 ms and interrupted
    resp = await _job_list_with_clock(db, large_batch.batch_id, _Clock(0, 0, 0, 0, 9.998), wide, **query)
    assert resp['pagination']['total_jobs'] is None
    assert resp['pagination']['page_end_reason'] == page_end_reason

    # nothing left: skipped
    resp = await _job_list_with_clock(db, large_batch.batch_id, _Clock(0, 0, 0, 0, 10.5), wide, **query)
    assert resp['pagination']['total_jobs'] is None

    # with time to spare it's exact
    resp = await _job_list_with_clock(db, large_batch.batch_id, _Clock(0), wide, **query)
    assert resp['pagination']['total_jobs'] == 0


async def test_dropped_connection_is_503_and_not_retried(db, large_batch):
    import asyncio  # pylint: disable=import-outside-toplevel

    import aiomysql  # pylint: disable=import-outside-toplevel
    from aiohttp import web  # pylint: disable=import-outside-toplevel

    from batch.front_end.job_list_api import job_list_response  # pylint: disable=import-outside-toplevel

    # slow enough to catch mid-query: ten never-matching text leaves over 20,000 jobs
    marker = 'killme'
    filter_ = {'or': [{'field': 'text', 'op': 'contains', 'value': f'{marker}{i}'} for i in range(10)]}
    query = {'job_group_ids': '0', 'recursive': 'true', 'filter': json.dumps(filter_)}
    request = asyncio.create_task(job_list_response(db, large_batch.batch_id, query, asyncio.Semaphore(1)))

    async def running_ids(cur):
        await cur.execute('SHOW FULL PROCESSLIST')
        return [r[0] for r in await cur.fetchall() if r[7] and f'{marker}0' in r[7] and 'PROCESSLIST' not in r[7]]

    admin = await aiomysql.connect(host='localhost', port=3306, user='root', password='pw')
    try:
        async with admin.cursor() as cur:
            killed = None
            for _ in range(400):
                ids_ = await running_ids(cur)
                if ids_:
                    killed = ids_[0]
                    await cur.execute(f'KILL {int(killed)}')
                    break
                assert not request.done(), 'finished before it could be killed'
                await asyncio.sleep(0.025)
            assert killed is not None, 'never saw the query running'

            # whatever error actually reaches the handler, the client sees a retryable 503
            with pytest.raises(web.HTTPServiceUnavailable) as e:
                await request
            assert e.value.headers['Retry-After']

            # run once: gear didn't retry it on another connection
            await asyncio.sleep(1)
            assert await running_ids(cur) == []
    finally:
        admin.close()

    # the pool recovers
    resp = await job_list(db, large_batch.batch_id, limit=1)
    assert len(resp['jobs']) == 1
