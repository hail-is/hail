import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

import pymysql
import pytest
from aiohttp import web

from batch.exceptions import QueryError
from batch.front_end import job_list_api
from batch.front_end.query.job_list import (
    _FIELDS,
    BACKWARD,
    BOUNDARY,
    FORWARD,
    LIMIT,
    MAX_JOB_ID,
    PARENT_EDGES,
    SCAN_SIZE,
    And,
    JobListLimits,
    Leaf,
    Or,
    PageEnd,
    PageLink,
    Window,
    cut_parent_edges,
    end_page,
    job_id_bounds,
    leaves,
    narrow_range,
    parse_filter,
    parse_include,
    parse_job_list_params,
    plan_window,
)
from batch.front_end.query.job_list_sql import (
    GroupFilter,
    JobGroupNotFound,
    JobListTimeout,
    Statement,
    attempts_statement,
    batch_range_statement,
    cost_per_hour_statement,
    count_statement,
    direct_group_range_statement,
    escape_like,
    groups_statement,
    jobs_statement,
    parent_edges_statement,
    recursive_group_range_statement,
    staged_pending_jobs_statement,
    subgroups_statement,
)

LIMITS = JobListLimits()


def f(x) -> str:
    return json.dumps(x)


def leaf(field, op, value=None, **kwargs):
    d = {'field': field, 'op': op, **kwargs}
    if op != 'exists':
        d['value'] = value
    return d


# Params


def test_params_defaults():
    p = parse_job_list_params({}, LIMITS)
    assert p.job_group_ids == (0,)
    assert p.recursive is False
    assert p.filter is None
    assert p.scan_direction == FORWARD
    assert p.scan_start_job_id is None
    assert p.limit == 50
    assert p.max_scan_size == LIMITS.max_scan_size
    assert not p.include


def test_params_values():
    p = parse_job_list_params(
        {
            'job_group_ids': '3,1,3',
            'recursive': 'true',
            'scan_direction': 'backward',
            'scan_start_job_id': '1000',
            'limit': '0',
            'max_scan_size': '1000',
            'include': 'cost,start_time',
            'filter': f(leaf('state', '=', 'Failed')),
        },
        LIMITS,
    )
    assert p.job_group_ids == (3, 1)
    assert p.recursive is True
    assert p.scan_direction == BACKWARD
    assert p.scan_start_job_id == 1000
    assert p.limit == 0
    assert p.max_scan_size == 1000
    assert p.include == ('start_time', 'cost')
    assert p.filter == Leaf('state', '=', 'Failed')


@pytest.mark.parametrize(
    'query',
    [
        {'job_group_ids': '-1'},
        {'job_group_ids': str(MAX_JOB_ID + 1)},
        {'job_group_ids': '1,,2'},
        {'job_group_ids': ' 1'},
        {'job_group_ids': '1_0'},
        {'job_group_ids': ','.join(str(i) for i in range(11))},
        {'recursive': 'yes'},
        {'scan_direction': 'up'},
        {'scan_start_job_id': str(MAX_JOB_ID + 1)},
        {'scan_start_job_id': '+5'},
        {'limit': '1001'},
        {'limit': '-1'},
        {'max_scan_size': '0'},
        {'max_scan_size': str(LIMITS.max_scan_size + 1)},
        {'include': 'bogus'},
        {'include': 'cost,'},
        {'filter': 'nope'},
    ],
)
def test_params_rejected(query):
    with pytest.raises(QueryError):
        parse_job_list_params(query, LIMITS)


def test_params_limits_are_parameters():
    small = JobListLimits(max_scan_size=10, max_job_group_ids=2)
    assert parse_job_list_params({}, small).max_scan_size == 10
    with pytest.raises(QueryError):
        parse_job_list_params({'max_scan_size': '11'}, small)
    with pytest.raises(QueryError):
        parse_job_list_params({'job_group_ids': '1,2,3'}, small)


def test_include_normalized():
    assert not parse_include(None)
    assert not parse_include('')
    assert parse_include('total_jobs,cost,start_time,cost') == ('start_time', 'cost', 'total_jobs')
    assert parse_include('attempts.cost_per_hour') == ('attempts', 'attempts.cost_per_hour')


# Filter


@pytest.mark.parametrize(
    'raw, expected',
    [
        (leaf('job_id', '>=', 5), Leaf('job_id', '>=', 5)),
        (leaf('job_id', 'in', [3, 1]), Leaf('job_id', 'in', (3, 1))),
        (leaf('state', 'in', ['Failed', 'Error']), Leaf('state', 'in', ('Failed', 'Error'))),
        (leaf('name', 'not_contains', 'x'), Leaf('name', 'not_contains', 'x')),
        (leaf('attribute', '=', '3', key='shard'), Leaf('attribute', '=', '3', 'shard')),
        (leaf('attribute', 'exists', key='shard'), Leaf('attribute', 'exists', None, 'shard')),
        (leaf('text', '=', 'a b'), Leaf('text', '=', 'a b')),
        (leaf('instance', 'contains', 'batch-worker'), Leaf('instance', 'contains', 'batch-worker')),
        (leaf('instance_collection', '=', 'standard'), Leaf('instance_collection', '=', 'standard')),
        (leaf('exit_code', 'in', [0, -1]), Leaf('exit_code', 'in', (0, -1))),
        (leaf('cost', '<', 1), Leaf('cost', '<', 1.0)),
        (leaf('cost', '>=', 0.5), Leaf('cost', '>=', 0.5)),
        (leaf('start_time', '>', 1727000000000), Leaf('start_time', '>', 1727000000000)),
        (leaf('end_time', '<', '2024-09-22T10:13:20Z'), Leaf('end_time', '<', 1727000000000)),
        (leaf('end_time', '<', '2024-09-22T06:13:20.001-04:00'), Leaf('end_time', '<', 1727000000001)),
        (leaf('duration', '<=', 0), Leaf('duration', '<=', 0)),
        (leaf('latest_attempt_duration', '>', 60000), Leaf('latest_attempt_duration', '>', 60000)),
    ],
)
def test_filter_leaves(raw, expected):
    assert parse_filter(f(raw), LIMITS) == expected


def test_filter_nesting():
    raw = {
        'and': [
            leaf('state', 'in', ['Failed', 'Error']),
            {'or': [leaf('name', 'contains', 'test_hail'), leaf('attribute', '=', '3', key='shard')]},
        ]
    }
    assert parse_filter(f(raw), LIMITS) == And((
        Leaf('state', 'in', ('Failed', 'Error')),
        Or((Leaf('name', 'contains', 'test_hail'), Leaf('attribute', '=', '3', 'shard'))),
    ))


def _nest(depth):
    node = leaf('state', '=', 'Failed')
    for i in range(depth):
        node = {'and' if i % 2 else 'or': [node]}
    return node


@pytest.mark.parametrize(
    'raw',
    [
        # structure
        [],
        'state',
        {},
        {'and': []},
        {'and': leaf('state', '=', 'Failed')},
        {'and': [], 'or': []},
        {'not': [leaf('state', '=', 'Failed')]},
        _nest(3),
        {'and': [leaf('job_id', '>', i) for i in range(11)]},
        {
            'or': [
                {'and': [leaf('job_id', '>', i) for i in range(6)]},
                {'and': [leaf('job_id', '<', i) for i in range(5)]},
            ]
        },
        # fields and operators
        leaf('bogus', '=', 1),
        leaf('state', 'contains', 'Fail'),
        leaf('job_id', '!=', 3),
        leaf('cost', '=', 1),
        leaf('start_time', '=', 1),
        leaf('exit_code', '<', 1),
        {'field': 'state', 'value': 'Failed'},
        # values
        leaf('state', '=', 'failed'),
        leaf('state', '=', 'bad'),
        leaf('state', 'in', []),
        leaf('state', 'in', 'Failed'),
        leaf('state', 'in', ['Failed'] * 101),
        leaf('job_id', '=', -1),
        leaf('job_id', '=', MAX_JOB_ID + 1),
        leaf('job_id', 'in', [1, MAX_JOB_ID + 1]),
        leaf('job_id', '=', '5'),
        leaf('job_id', '=', 5.0),
        leaf('job_id', '=', True),
        leaf('exit_code', '=', None),
        leaf('exit_code', '=', 2**31),
        leaf('cost', '<', '1'),
        leaf('cost', '<', True),
        leaf('name', '=', 5),
        leaf('name', '=', None),
        leaf('name', '=', ['a']),
        leaf('name', '=', {'a': 1}),
        leaf('duration', '<', -1),
        leaf('duration', '<', 1.5),
        # times
        leaf('start_time', '>', '2024-09-22T10:13:20'),
        leaf('start_time', '>', '2024-09-22'),
        leaf('start_time', '>', 'yesterday'),
        leaf('start_time', '>', '1969-12-31T00:00:00Z'),
        leaf('start_time', '>', -1),
        # keys
        leaf('attribute', '=', '3'),
        leaf('attribute', '=', '3', key=5),
        {**leaf('attribute', 'exists', key='shard'), 'value': 'x'},
        leaf('name', '=', 'x', key='shard'),
        {**leaf('state', '=', 'Failed'), 'extra': 1},
    ],
)
def test_filter_rejected(raw):
    with pytest.raises(QueryError):
        parse_filter(f(raw), LIMITS)


@pytest.mark.parametrize(
    's',
    [
        'not json',
        '{"field": "state", "op": "=", "value": "Failed"',
        '{"field": "state", "field": "state", "op": "=", "value": "Failed"}',
        '{"field": "cost", "op": "<", "value": NaN}',
        '{"field": "cost", "op": "<", "value": Infinity}',
        '[' * 1000 + ']' * 1000,
        '{"and":' * 200 + '[]' + '}' * 200,
    ],
    ids=['not_json', 'truncated', 'duplicate_key', 'nan', 'infinity', 'deep_array', 'deep_object'],
)
def test_filter_bad_json(s):
    assert len(s.encode('utf-8')) <= LIMITS.max_filter_bytes
    with pytest.raises(QueryError):
        parse_filter(s, LIMITS)


def test_filter_limits_allowed_at_the_edge():
    ten = {'and': [leaf('job_id', '>', i) for i in range(10)]}
    parsed = parse_filter(f(ten), LIMITS)
    assert isinstance(parsed, And) and len(parsed.children) == 10
    deepest = {'and': [{'or': [leaf('state', '=', 'Failed'), leaf('state', '=', 'Error')]}]}
    parse_filter(f(deepest), LIMITS)
    parse_filter(f(leaf('job_id', 'in', list(range(100)))), LIMITS)


def test_filter_bytes_are_utf8():
    prefix = f(leaf('name', 'contains', ''))[:-2]
    fits = prefix + 'x' * (LIMITS.max_filter_bytes - len(prefix) - 2) + '"}'
    assert len(fits) == LIMITS.max_filter_bytes
    parse_filter(fits, LIMITS)
    # the same number of characters, but each is 3 bytes in UTF-8
    over = prefix + '名' * (LIMITS.max_filter_bytes - len(prefix) - 2) + '"}'
    assert len(over) == LIMITS.max_filter_bytes
    with pytest.raises(QueryError):
        parse_filter(over, LIMITS)


# job_id narrowing


def test_job_id_bounds():
    def bounds(raw):
        return job_id_bounds(parse_filter(f(raw), LIMITS))

    assert job_id_bounds(None) == (0, MAX_JOB_ID)
    assert bounds(leaf('job_id', '=', 7)) == (7, 7)
    assert bounds(leaf('job_id', 'in', [9, 3, 5])) == (3, 9)
    assert bounds(leaf('job_id', '<', 10)) == (0, 9)
    assert bounds(leaf('job_id', '<=', 10)) == (0, 10)
    assert bounds(leaf('job_id', '>', 10)) == (11, MAX_JOB_ID)
    assert bounds({'and': [leaf('job_id', '>=', 900), leaf('job_id', '<', 1000), leaf('state', '=', 'Failed')]}) == (
        900,
        999,
    )
    assert bounds({'and': [leaf('job_id', '>=', 900), leaf('job_id', '<=', 100)]}) == (900, 100)
    # under an `or`, at the top or nested: no narrowing
    assert bounds({'or': [leaf('job_id', '=', 7), leaf('state', '=', 'Failed')]}) == (0, MAX_JOB_ID)
    assert bounds({'and': [{'or': [leaf('job_id', '=', 7), leaf('job_id', '=', 9)]}]}) == (0, MAX_JOB_ID)


def test_narrow_range():
    assert narrow_range(None, None, (0, 10)) == (None, None)
    assert narrow_range(1, 100, (0, MAX_JOB_ID)) == (1, 100)
    assert narrow_range(1, 100, (50, 60)) == (50, 60)
    assert narrow_range(1, 100, (900, 1000)) == (None, None)
    assert narrow_range(1, 100, (900, 100)) == (None, None)
    assert narrow_range(1, 100, (0, 0)) == (None, None)


# Windows


def test_window_defaults_to_range_ends():
    assert plan_window(10, 100, FORWARD, None, 50) == Window(FORWARD, 10, 10, 59, 10, 100)
    assert plan_window(10, 100, BACKWARD, None, 50) == Window(BACKWARD, 100, 51, 100, 10, 100)


def test_window_anchored_and_intersected():
    # a fixed block starting below the range keeps its own end, so it can't overlap the next block
    assert plan_window(5437, 9000, FORWARD, 5001, 1000) == Window(FORWARD, 5001, 5437, 6000, 5437, 9000)
    assert plan_window(1, 5437, BACKWARD, 6000, 1000) == Window(BACKWARD, 6000, 5001, 5437, 1, 5437)


def test_window_empty_range():
    assert plan_window(None, None, FORWARD, None, 50) == PageEnd(BOUNDARY, None, None)
    assert plan_window(None, None, BACKWARD, 7, 50) == PageEnd(BOUNDARY, None, None)


def test_window_past_the_range():
    assert plan_window(10, 100, FORWARD, 101, 50) == PageEnd(BOUNDARY, None, PageLink(100, BACKWARD))
    assert plan_window(10, 100, BACKWARD, 9, 50) == PageEnd(BOUNDARY, PageLink(10, FORWARD), None)


def test_window_before_the_range():
    assert plan_window(1000, 2000, FORWARD, 1, 999) == PageEnd(SCAN_SIZE, PageLink(1000, FORWARD), None)
    assert plan_window(1000, 2000, BACKWARD, 3000, 1000) == PageEnd(SCAN_SIZE, None, PageLink(2000, BACKWARD))
    # touching the range is a scan, not a jump
    assert plan_window(1000, 2000, FORWARD, 1, 1000) == Window(FORWARD, 1, 1000, 1000, 1000, 2000)


# Page ends


def w(direction, start, lo, hi, range_min=1, range_max=1000):
    return Window(direction, start, lo, hi, range_min, range_max)


def test_end_limit_with_lookahead():
    page = list(range(1, 51))
    assert end_page(w(FORWARD, 1, 1, 1000), page, 50, True, 50) == PageEnd(LIMIT, PageLink(51, FORWARD), None)
    page = list(range(1000, 950, -1))
    assert end_page(w(BACKWARD, 1000, 1, 1000), page, 50, True, 50) == PageEnd(LIMIT, None, PageLink(950, BACKWARD))


def test_end_limit_filled_at_window_end():
    page = list(range(51, 101))
    assert end_page(w(FORWARD, 51, 51, 100), page, 50, False, 50) == PageEnd(
        LIMIT, PageLink(101, FORWARD), PageLink(50, BACKWARD)
    )


def test_end_boundary_over_limit():
    # exactly `limit` rows left before the edge: the lookahead finds nothing, so it's the boundary
    page = list(range(951, 1001))
    assert end_page(w(FORWARD, 951, 951, 1000), page, 50, False, 50) == PageEnd(BOUNDARY, None, PageLink(950, BACKWARD))
    page = list(range(50, 0, -1))
    assert end_page(w(BACKWARD, 50, 1, 50), page, 50, False, 50) == PageEnd(BOUNDARY, PageLink(51, FORWARD), None)


def test_end_scan_size_and_boundary():
    assert end_page(w(FORWARD, 101, 101, 200), [150], 50, False, 1) == PageEnd(
        SCAN_SIZE, PageLink(201, FORWARD), PageLink(100, BACKWARD)
    )
    assert end_page(w(BACKWARD, 200, 101, 200), [], 50, False, 0) == PageEnd(
        SCAN_SIZE, PageLink(201, FORWARD), PageLink(100, BACKWARD)
    )
    assert end_page(w(FORWARD, 901, 901, 1000), [950], 50, False, 1) == PageEnd(BOUNDARY, None, PageLink(900, BACKWARD))


def test_end_other_link_null_at_range_edge():
    assert end_page(w(FORWARD, 1, 1, 100), [], 50, False, 0).previous_page is None
    # a block starting below the range
    assert end_page(w(FORWARD, 1, 5, 100, range_min=5), [], 50, False, 0).previous_page is None
    assert end_page(w(BACKWARD, 1000, 901, 1000), [], 50, False, 0).next_page is None


def test_end_parent_edges_first():
    page = [1, 2, 3]
    assert end_page(w(FORWARD, 1, 1, 1000), page, 3, True, 2) == PageEnd(PARENT_EDGES, PageLink(3, FORWARD), None)
    # even at the boundary, so the cut jobs stay reachable
    page = [998, 999, 1000]
    assert end_page(w(FORWARD, 998, 998, 1000), page, 50, False, 1) == PageEnd(
        PARENT_EDGES, PageLink(999, FORWARD), PageLink(997, BACKWARD)
    )
    page = [1000, 999, 998]
    assert end_page(w(BACKWARD, 1000, 1, 1000), page, 3, True, 2) == PageEnd(
        PARENT_EDGES, None, PageLink(998, BACKWARD)
    )


# Parent edges


def test_parent_edges_under_budget():
    edges = cut_parent_edges([1, 2, 3], [(1, 5), (1, 4), (3, 1)], 3)
    assert edges.n_kept == 3
    assert edges.parent_ids == {1: [4, 5], 2: [], 3: [1]}
    assert edges.first_truncated is False


def test_parent_edges_cut_before_the_owner():
    # row 4 (budget 3 + 1) belongs to job 3: jobs 1 and 2 (no parents) are kept
    edges = cut_parent_edges([1, 2, 3, 4], [(1, 7), (1, 8), (3, 1), (3, 2)], 3)
    assert edges.n_kept == 2
    assert edges.parent_ids == {1: [7, 8], 2: []}
    assert edges.first_truncated is False


def test_parent_edges_lone_job_truncated():
    edges = cut_parent_edges([5, 6], [(5, 1), (5, 2), (5, 3), (5, 4)], 3)
    assert edges.n_kept == 1
    assert edges.parent_ids == {5: [1, 2, 3]}
    assert edges.first_truncated is True


def test_parent_edges_exactly_at_budget_not_truncated():
    edges = cut_parent_edges([5, 6], [(5, 1), (5, 2), (5, 3), (6, 1)], 3)
    assert edges.n_kept == 1
    assert edges.parent_ids == {5: [1, 2, 3]}
    assert edges.first_truncated is False


def test_parent_edges_backward_order():
    edges = cut_parent_edges([9, 8, 7], [(9, 3), (9, 2), (8, 5), (7, 1)], 3)
    assert edges.n_kept == 2
    assert edges.parent_ids == {9: [2, 3], 8: [5]}


# Paging property: a model server built from these functions, followed link by link


def _serve(
    matching: Sequence[int],
    parents: Dict[int, List[int]],
    range_min: Optional[int],
    range_max: Optional[int],
    link: Tuple[Optional[int], str],
    limit: int,
    max_scan_size: int,
    max_edges: Optional[int],
) -> Tuple[List[int], PageEnd]:
    start, direction = link
    plan = plan_window(range_min, range_max, direction, start, max_scan_size)
    if isinstance(plan, PageEnd):
        return [], plan
    in_window = [i for i in matching if plan.lo <= i <= plan.hi]
    if direction == BACKWARD:
        in_window.reverse()
    found = in_window[: limit + 1]
    page = found[:limit]
    n_kept = len(page)
    if max_edges is not None:
        rows = [(j, p) for j in page for p in sorted(parents[j], reverse=direction == BACKWARD)]
        n_kept = cut_parent_edges(page, rows[: max_edges + 1], max_edges).n_kept
    return sorted(page[:n_kept]), end_page(plan, page, limit, len(found) > limit, n_kept)


def _follow(serve, first: Tuple[Optional[int], str]) -> List[int]:
    seen: List[int] = []
    rows, end = serve(first)
    seen += rows
    for attr in ('next_page', 'previous_page'):
        link = getattr(end, attr)
        for _ in range(10_000):
            if link is None:
                break
            rows, page_end = serve((link.scan_start_job_id, link.scan_direction))
            assert rows == sorted(rows)
            seen += rows
            link = getattr(page_end, attr)
        else:
            raise AssertionError('paging did not terminate')
    return seen


@pytest.mark.parametrize('seed', range(300))
def test_paging_property(seed):
    rng = random.Random(seed)
    if rng.random() < 0.1:
        range_min, range_max = None, None
        all_ids: List[int] = []
    else:
        range_min = rng.randint(1, 200)
        range_max = range_min + rng.randint(0, 300)
        all_ids = list(range(range_min, range_max + 1))
    density = rng.choice([1.0, 0.5, 0.05, 0.0])
    matching = [i for i in all_ids if rng.random() < density]
    parents = {i: rng.sample(range(1, 50), rng.choice([0, 0, 1, 3, 8])) for i in matching}
    limit = rng.randint(1, 20)
    max_scan_size = rng.randint(1, 80)
    max_edges = rng.choice([None, 4, 10])

    def serve(link):
        return _serve(matching, parents, range_min, range_max, link, limit, max_scan_size, max_edges)

    start = rng.choice([None, rng.randint(0, 600)])
    direction = rng.choice([FORWARD, BACKWARD])
    seen = _follow(serve, (start, direction))
    assert len(seen) == len(set(seen)), 'a job came back twice'
    assert sorted(seen) == matching


# SQL builders, without a database

BATCH_ID = 424242424
GROUP_ID = 31337
JOB_ID = 987654321
STRINGS = ["SENTINEL_a", "'; DROP TABLE jobs; --", '%_\\', '名前"', 'a`b']
WIDE = JobListLimits(max_filter_leaves=100, max_filter_bytes=100_000)


def _all_leaves_filters() -> List[str]:
    """Filters that between them use every field and operator, with distinctive values."""
    s0, s1, s2, s3, s4 = STRINGS
    per_field = [
        leaf('job_id', '=', JOB_ID),
        leaf('job_id', 'in', [JOB_ID, JOB_ID - 1]),
        *[leaf('job_id', op, JOB_ID) for op in ('<', '<=', '>', '>=')],
        leaf('state', '=', 'Failed'),
        leaf('state', '!=', 'Success'),
        leaf('state', 'in', ['Error', 'Cancelled']),
        *[leaf('name', op, s) for op, s in [('=', s0), ('!=', s1), ('contains', s2), ('not_contains', s3)]],
        *[
            leaf('attribute', op, s, key=s4)
            for op, s in [('=', s0), ('!=', s1), ('contains', s2), ('not_contains', s3)]
        ],
        leaf('attribute', 'exists', key=s1),
        leaf('text', 'contains', s2),
        leaf('text', '=', s1),
        leaf('instance', '=', s0),
        leaf('instance', 'contains', s3),
        leaf('instance_collection', '=', s4),
        leaf('exit_code', '=', -77777),
        leaf('exit_code', '!=', -77778),
        leaf('exit_code', 'in', [-77779, -77780]),
        *[leaf('cost', op, 12345.678) for op in ('<', '<=', '>', '>=')],
        *[leaf('start_time', op, '2031-07-05T01:02:03.004Z') for op in ('<', '<=', '>', '>=')],
        *[leaf('end_time', op, 1940979723005) for op in ('<', '<=', '>', '>=')],
        *[leaf('duration', op, 86400017) for op in ('<', '<=', '>', '>=')],
        *[leaf('latest_attempt_duration', op, 86400019) for op in ('<', '<=', '>', '>=')],
    ]
    return [
        f({'and': per_field}),
        f({'or': per_field}),
        f({'and': [{'or': per_field[:20]}, {'or': per_field[20:]}]}),
    ]


def test_all_leaves_filters_cover_every_field_and_op():
    # So a new field or operator can't skip the values-only-in-args checks below.
    used = {(lf.field, lf.op) for raw in _all_leaves_filters() for lf in leaves(parse_filter(raw, WIDE))}
    assert used == {(field, op) for field, (ops, _) in _FIELDS.items() for op in ops}


ALL_INCLUDES = parse_include(','.join(['start_time', 'end_time', 'latest_attempt_duration', 'exit_code', 'cost']))


def _statements() -> List[Statement]:
    stmts = [
        groups_statement(BATCH_ID, [GROUP_ID, GROUP_ID + 1]),
        batch_range_statement(BATCH_ID),
        direct_group_range_statement(BATCH_ID, [GROUP_ID, GROUP_ID + 1]),
        staged_pending_jobs_statement(BATCH_ID, [GROUP_ID]),
        subgroups_statement(BATCH_ID, [GROUP_ID], 10001),
        recursive_group_range_statement(BATCH_ID, [GROUP_ID, GROUP_ID + 1]),
        attempts_statement(BATCH_ID, [JOB_ID, JOB_ID - 1]),
        cost_per_hour_statement(BATCH_ID, [JOB_ID]),
        parent_edges_statement(BATCH_ID, [JOB_ID], BACKWARD, 10001),
    ]
    groups = [None, GroupFilter((GROUP_ID,), False), GroupFilter((GROUP_ID, GROUP_ID + 1), True)]
    for raw in [None, *_all_leaves_filters()]:
        filter_ = None if raw is None else parse_filter(raw, WIDE)
        for group_filter in groups:
            for direction in (FORWARD, BACKWARD):
                stmts.append(
                    jobs_statement(BATCH_ID, JOB_ID, JOB_ID + 3, direction, 51, group_filter, filter_, ALL_INCLUDES)
                )
            stmts.append(count_statement(BATCH_ID, JOB_ID, JOB_ID + 3, group_filter, filter_))
    return stmts


STATEMENTS = _statements()
STATEMENT_IDS = [s.name for s in STATEMENTS]


@pytest.mark.parametrize('stmt', STATEMENTS, ids=STATEMENT_IDS)
def test_statement_hint_first(stmt: Statement):
    sql = stmt.sql(7)
    assert sql.startswith('SELECT /*+ MAX_EXECUTION_TIME(7) */ ')
    # anywhere else MySQL ignores it with only a warning
    assert sql.count('/*+') == 1


@pytest.mark.parametrize('stmt', STATEMENTS, ids=STATEMENT_IDS)
def test_statement_values_only_in_args(stmt: Statement):
    # Substitution uses Python's %, so a stray % in the text would break it, and each value needs a placeholder.
    assert stmt.body.count('%s') == len(stmt.args)
    assert '%' not in stmt.body.replace('%s', '')
    distinctive = [BATCH_ID, GROUP_ID, JOB_ID, -77777, 12345.678, 1940979723005, 86400017, *STRINGS]
    for v in distinctive:
        assert str(v) not in stmt.body, v
    for s in STRINGS:
        for part in s.split():
            assert part not in stmt.body or part in ('--',), part


def test_statement_args_carry_the_values():
    stmt = jobs_statement(
        BATCH_ID, 1, 2, FORWARD, 51, GroupFilter((GROUP_ID,), True), parse_filter(_all_leaves_filters()[0], WIDE), ()
    )
    args = [str(a) for a in stmt.args]
    for s in STRINGS:
        assert s in args or f'%{escape_like(s)}%' in args, s
    for v in (BATCH_ID, GROUP_ID, JOB_ID, -77777, 12345.678, 1940979723005, 86400017):
        assert str(v) in args, v


@pytest.mark.parametrize('stmt', STATEMENTS, ids=STATEMENT_IDS)
def test_statement_subqueries_correlated(stmt: Statement):
    body = stmt.body
    for alias, table in stmt.aliases.items():
        assert body.count(f'{table} AS {alias}') == 1, alias
        if alias in ('upd', 'anc', 'j', 'ar', 'r', 'name_attr', 'cost_resources', 'staging'):
            continue
        # every subquery on another per-batch table is tied to the outer job
        assert f'{alias}.batch_id = jobs.batch_id' in body, alias
        column = 'job_group_id' if alias == 'grp' else 'job_id'
        assert f'{alias}.{column} = jobs.{column}' in body, alias


@pytest.mark.parametrize('stmt', STATEMENTS, ids=STATEMENT_IDS)
def test_statement_derived_tables_once(stmt: Statement):
    for derived in ('attempt_summary', 'cost_t'):
        assert stmt.body.count(f') AS {derived} ON TRUE') <= 1, derived


@pytest.mark.parametrize('ms', [0, -1, 1.5, True, '5', None])
def test_statement_bad_time_limit(ms):
    with pytest.raises(ValueError):
        batch_range_statement(BATCH_ID).sql(ms)


def test_join_order_follows_the_filter():
    time_leaf = parse_filter(f(leaf('end_time', '>', 1)), LIMITS)
    cost_leaf = parse_filter(f(leaf('cost', '>', 1)), LIMITS)
    includes = parse_include('start_time,cost')

    body = jobs_statement(BATCH_ID, 1, 2, FORWARD, 51, None, time_leaf, includes).body
    assert body.index('AS attempt_summary') < min(body.index('AS cost_t'), body.index('AS name_attr'))

    body = jobs_statement(BATCH_ID, 1, 2, FORWARD, 51, None, cost_leaf, includes).body
    assert body.index('AS cost_t') < body.index('AS attempt_summary')

    name_leaf = parse_filter(f(leaf('name', '=', 'x')), LIMITS)
    body = jobs_statement(BATCH_ID, 1, 2, FORWARD, 51, None, name_leaf, includes).body
    assert body.index('AS name_attr') < body.index('AS attempt_summary')


def test_only_needed_fragments():
    body = jobs_statement(BATCH_ID, 1, 2, FORWARD, 51, None, None, ()).body
    assert 'attempt_summary' not in body and 'cost_t' not in body and 'status' not in body
    state_leaf = parse_filter(f(leaf('state', '=', 'Failed')), LIMITS)
    body = count_statement(BATCH_ID, 1, 2, None, state_leaf).body
    assert 'name_attr' not in body and 'attempt_summary' not in body and 'cost_t' not in body


def test_escape_like():
    assert escape_like('a_b%c\\d') == 'a\\_b\\%c\\\\d'
    assert escape_like('plain') == 'plain'


# The handler: error mapping and the concurrency limit, with the query layer stubbed out


def _raising(exc: BaseException):
    async def get_job_list(*args, **kwargs):  # pylint: disable=unused-argument
        raise exc

    return get_job_list


def _with_context(outer: BaseException, inner: BaseException) -> BaseException:
    outer.__context__ = inner
    return outer


async def _respond(monkeypatch, exc: BaseException, query=None, semaphore=None):
    import asyncio  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(job_list_api, 'get_job_list', _raising(exc))
    semaphore = semaphore or asyncio.Semaphore(1)
    try:
        await job_list_api.job_list_response(None, 1, query or {}, semaphore)  # type: ignore
    finally:
        # released whatever happened
        assert not semaphore.locked()


@pytest.mark.parametrize(
    'exc, status',
    [
        (QueryError('bad'), 400),
        (JobGroupNotFound(), 404),
        (JobListTimeout(), 503),
        (pymysql.err.OperationalError(2013, 'Lost connection'), 503),
        (pymysql.err.OperationalError(1040, 'Too many connections'), 503),
        (pymysql.err.OperationalError(3024, 'maximum statement execution time exceeded'), 503),
        (pymysql.err.InterfaceError(0, ''), 503),
        (pymysql.err.InternalError(1205, 'Lock wait timeout'), 503),
        # rolling back on a dead connection replaces the original error
        (_with_context(RuntimeError('rollback failed'), pymysql.err.OperationalError(2013, 'Lost')), 503),
    ],
)
async def test_handler_maps_errors(monkeypatch, exc, status):
    with pytest.raises(web.HTTPException) as e:
        await _respond(monkeypatch, exc)
    assert e.value.status == status
    if status == 503:
        assert e.value.headers['Retry-After'] == str(job_list_api.RETRY_AFTER_SECS)


@pytest.mark.parametrize('exc', [ValueError('a bug'), pymysql.err.InternalError(1064, 'syntax'), KeyError('x')])
async def test_handler_leaves_other_errors_alone(monkeypatch, exc):
    with pytest.raises(type(exc)):
        await _respond(monkeypatch, exc)


async def test_handler_timeout_suggests_narrowing_a_filter(monkeypatch):
    with pytest.raises(web.HTTPServiceUnavailable) as e:
        await _respond(monkeypatch, JobListTimeout(), {'filter': f(leaf('state', '=', 'Failed'))})
    assert 'narrow the filter' in (e.value.text or '')
    with pytest.raises(web.HTTPServiceUnavailable) as e:
        await _respond(monkeypatch, JobListTimeout())
    assert 'narrow the filter' not in (e.value.text or '')


async def test_handler_bad_params_never_wait_for_the_database(monkeypatch):
    import asyncio  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(job_list_api, 'get_job_list', _raising(AssertionError('should not be called')))
    held = asyncio.Semaphore(1)
    await held.acquire()
    with pytest.raises(web.HTTPBadRequest):
        await job_list_api.job_list_response(None, 1, {'limit': 'lots'}, held, queue_wait_secs=0.01)  # type: ignore


async def test_handler_waits_for_a_slot_then_runs(monkeypatch):
    import asyncio  # pylint: disable=import-outside-toplevel

    async def ok(*args, **kwargs):  # pylint: disable=unused-argument
        return {'jobs': []}

    monkeypatch.setattr(job_list_api, 'get_job_list', ok)
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    asyncio.get_running_loop().call_later(0.05, semaphore.release)
    assert await job_list_api.job_list_response(None, 1, {}, semaphore, queue_wait_secs=2) == {'jobs': []}  # type: ignore
    assert not semaphore.locked()


async def test_handler_busy_after_the_queue_wait(monkeypatch):
    import asyncio  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(job_list_api, 'get_job_list', _raising(AssertionError('should not be called')))
    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    with pytest.raises(web.HTTPServiceUnavailable) as e:
        await job_list_api.job_list_response(None, 1, {}, semaphore, queue_wait_secs=0.05)  # type: ignore
    assert e.value.headers['Retry-After'] == str(job_list_api.RETRY_AFTER_SECS)
    # the slot it never got is still the holder's
    assert semaphore.locked()
    semaphore.release()
    assert not semaphore.locked()
