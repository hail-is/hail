import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

from ...exceptions import QueryError
from .query import JobState

MAX_JOB_ID = 2**31 - 1
MAX_LIMIT = 1000
DEFAULT_LIMIT = 50

FORWARD = 'forward'
BACKWARD = 'backward'

LIMIT = 'limit'
SCAN_SIZE = 'scan_size'
PARENT_EDGES = 'parent_edges'
BOUNDARY = 'boundary'

# The order the response echoes them in.
INCLUDES = (
    'start_time',
    'end_time',
    'latest_attempt_duration',
    'exit_code',
    'attempts',
    'attempts.cost_per_hour',
    'parent_ids',
    'cost',
    'total_jobs',
)

JOB_STATES = tuple(s.value for s in JobState)

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class JobListLimits:
    max_scan_size: int = 50_000
    total_jobs_count_max: int = 10_000
    parent_edge_page_max: int = 10_000
    group_range_lookup_max_jobs: int = 20_000
    group_range_lookup_max_subgroups: int = 10_000
    max_job_group_ids: int = 10
    max_filter_leaves: int = 10
    max_filter_depth: int = 2
    max_in_values: int = 100
    max_filter_bytes: int = 2048
    query_time_limit_ms: int = 10_000


LeafValue = Union[None, str, int, float, Tuple[str, ...], Tuple[int, ...]]


@dataclass(frozen=True)
class Leaf:
    field: str
    op: str
    # Normalized: times are ms, `in` values are a tuple, `exists` has None.
    value: LeafValue
    key: Optional[str] = None


@dataclass(frozen=True)
class And:
    children: Tuple['FilterNode', ...]


@dataclass(frozen=True)
class Or:
    children: Tuple['FilterNode', ...]


FilterNode = Union[Leaf, And, Or]


@dataclass(frozen=True)
class JobListParams:
    job_group_ids: Tuple[int, ...]
    recursive: bool
    filter: Optional[FilterNode]
    scan_direction: str
    scan_start_job_id: Optional[int]
    limit: int
    max_scan_size: int
    include: Tuple[str, ...]


@dataclass(frozen=True)
class PageLink:
    scan_start_job_id: int
    scan_direction: str

    def to_json(self) -> Dict[str, Any]:
        return {'scan_start_job_id': self.scan_start_job_id, 'scan_direction': self.scan_direction}


@dataclass(frozen=True)
class PageEnd:
    reason: Optional[str]
    next_page: Optional[PageLink]
    previous_page: Optional[PageLink]


@dataclass(frozen=True)
class Window:
    """The job ids one request scans: the anchored window intersected with the scan range."""

    direction: str
    start: int
    lo: int
    hi: int
    range_min: int
    range_max: int

    @property
    def reaches_boundary(self) -> bool:
        if self.direction == FORWARD:
            return self.hi >= self.range_max
        return self.lo <= self.range_min


_UINT_RE = re.compile(r'[0-9]+')


def _parse_uint(name: str, s: str, lo: int, hi: int) -> int:
    # int() would also accept whitespace, signs and underscores.
    if not _UINT_RE.fullmatch(s):
        raise QueryError(f'{name}: expected an integer, got {s!r}')
    v = int(s)
    if not lo <= v <= hi:
        raise QueryError(f'{name}: must be between {lo} and {hi}, got {v}')
    return v


def _parse_bool(name: str, s: str) -> bool:
    if s in ('true', 'True', '1'):
        return True
    if s in ('false', 'False', '0'):
        return False
    raise QueryError(f'{name}: expected true or false, got {s!r}')


def parse_include(s: Optional[str]) -> Tuple[str, ...]:
    if not s:
        return ()
    requested = set()
    for item in s.split(','):
        if item not in INCLUDES:
            raise QueryError(f'include: unknown value {item!r}; expected any of {", ".join(INCLUDES)}')
        requested.add(item)
    if 'attempts.cost_per_hour' in requested:
        requested.add('attempts')
    return tuple(i for i in INCLUDES if i in requested)


def parse_job_list_params(query: Mapping[str, str], limits: JobListLimits) -> JobListParams:
    job_group_ids_s = query.get('job_group_ids')
    if job_group_ids_s is None:
        job_group_ids: Tuple[int, ...] = (0,)
    else:
        ids = [_parse_uint('job_group_ids', s, 0, MAX_JOB_ID) for s in job_group_ids_s.split(',')]
        job_group_ids = tuple(dict.fromkeys(ids))
        if len(job_group_ids) > limits.max_job_group_ids:
            raise QueryError(f'job_group_ids: at most {limits.max_job_group_ids} groups, got {len(job_group_ids)}')

    recursive_s = query.get('recursive')
    recursive = False if recursive_s is None else _parse_bool('recursive', recursive_s)

    scan_direction = query.get('scan_direction', FORWARD)
    if scan_direction not in (FORWARD, BACKWARD):
        raise QueryError(f'scan_direction: expected forward or backward, got {scan_direction!r}')

    start_s = query.get('scan_start_job_id')
    scan_start_job_id = None if start_s is None else _parse_uint('scan_start_job_id', start_s, 0, MAX_JOB_ID)

    limit_s = query.get('limit')
    limit = DEFAULT_LIMIT if limit_s is None else _parse_uint('limit', limit_s, 0, MAX_LIMIT)

    max_scan_size_s = query.get('max_scan_size')
    max_scan_size = (
        limits.max_scan_size
        if max_scan_size_s is None
        else _parse_uint('max_scan_size', max_scan_size_s, 1, limits.max_scan_size)
    )

    filter_s = query.get('filter')
    filter_ = None if filter_s is None else parse_filter(filter_s, limits)

    return JobListParams(
        job_group_ids=job_group_ids,
        recursive=recursive,
        filter=filter_,
        scan_direction=scan_direction,
        scan_start_job_id=scan_start_job_id,
        limit=limit,
        max_scan_size=max_scan_size,
        include=parse_include(query.get('include')),
    )


# Filter


_COMPARISONS = ('<', '<=', '>', '>=')


def _int_value(v: Any, lo: int, hi: int) -> int:
    # bool is an int subclass, and JSON true/false must not pass as 1/0.
    if not isinstance(v, int) or isinstance(v, bool):
        raise QueryError(f'filter: expected an integer, got {json.dumps(v)}')
    if not lo <= v <= hi:
        raise QueryError(f'filter: {v} is outside {lo}..{hi}')
    return v


def _job_id_value(v: Any) -> int:
    return _int_value(v, 0, MAX_JOB_ID)


def _exit_code_value(v: Any) -> int:
    return _int_value(v, -(2**31), 2**31 - 1)


def _ms_value(v: Any) -> int:
    return _int_value(v, 0, 2**63 - 1)


def _time_value(v: Any) -> int:
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v)
        except ValueError as e:
            raise QueryError(f'filter: expected an ISO-8601 time, got {json.dumps(v)}') from e
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise QueryError(f'filter: time {json.dumps(v)} needs "Z" or an explicit offset')
        return _ms_value((dt - _EPOCH) // timedelta(milliseconds=1))
    return _ms_value(v)


def _cost_value(v: Any) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise QueryError(f'filter: expected a number, got {json.dumps(v)}')
    return float(v)


def _str_value(v: Any) -> str:
    if not isinstance(v, str):
        raise QueryError(f'filter: expected a string, got {json.dumps(v)}')
    return v


def _state_value(v: Any) -> str:
    if v not in JOB_STATES:
        raise QueryError(f'filter: unknown state {json.dumps(v)}; expected one of {", ".join(JOB_STATES)}')
    return v


# field -> (operators, value parser)
_FIELDS = {
    'job_id': (('=', 'in', *_COMPARISONS), _job_id_value),
    'state': (('=', '!=', 'in'), _state_value),
    'name': (('=', '!=', 'contains', 'not_contains'), _str_value),
    'attribute': (('=', '!=', 'contains', 'not_contains', 'exists'), _str_value),
    'text': (('contains', '='), _str_value),
    'instance': (('=', 'contains'), _str_value),
    'instance_collection': (('=',), _str_value),
    'exit_code': (('=', '!=', 'in'), _exit_code_value),
    'cost': (_COMPARISONS, _cost_value),
    'start_time': (_COMPARISONS, _time_value),
    'end_time': (_COMPARISONS, _time_value),
    'duration': (_COMPARISONS, _ms_value),
    'latest_attempt_duration': (_COMPARISONS, _ms_value),
}


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    d = {}
    for k, v in pairs:
        if k in d:
            raise QueryError(f'filter: duplicate key {json.dumps(k)}')
        d[k] = v
    return d


def _reject_constant(c: str):
    raise QueryError(f'filter: {c} is not allowed')


def parse_filter(s: str, limits: JobListLimits) -> FilterNode:
    n_bytes = len(s.encode('utf-8'))
    if n_bytes > limits.max_filter_bytes:
        raise QueryError(f'filter: at most {limits.max_filter_bytes} bytes of JSON, got {n_bytes}')
    try:
        raw = json.loads(s, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as e:
        raise QueryError('filter: invalid JSON') from e
    n_leaves = 0

    def node(x: Any, depth: int) -> FilterNode:
        nonlocal n_leaves
        if not isinstance(x, dict):
            raise QueryError(f'filter: expected an object, got {json.dumps(x)}')
        if 'field' in x:
            n_leaves += 1
            if n_leaves > limits.max_filter_leaves:
                raise QueryError(f'filter: at most {limits.max_filter_leaves} leaves')
            return _leaf(x, limits)
        if len(x) != 1 or next(iter(x)) not in ('and', 'or'):
            raise QueryError(f'filter: expected {{"and": [...]}}, {{"or": [...]}} or a leaf, got keys {sorted(x)}')
        ((kind, children),) = x.items()
        if depth >= limits.max_filter_depth:
            raise QueryError(f'filter: at most {limits.max_filter_depth} nested levels of and/or')
        if not isinstance(children, list) or not children:
            raise QueryError(f'filter: "{kind}" needs a non-empty list')
        parsed = tuple(node(c, depth + 1) for c in children)
        return And(parsed) if kind == 'and' else Or(parsed)

    return node(raw, 0)


def _leaf(x: Dict[str, Any], limits: JobListLimits) -> Leaf:
    field = x['field']
    if field not in _FIELDS:
        raise QueryError(f'filter: unknown field {json.dumps(field)}')
    ops, parse_value = _FIELDS[field]
    op = x.get('op')
    if not isinstance(op, str) or op not in ops:
        raise QueryError(f'filter: {field} supports {", ".join(ops)}, got {json.dumps(op)}')

    allowed_keys = {'field', 'op'}
    key = None
    if field == 'attribute':
        allowed_keys.add('key')
        key = x.get('key')
        if not isinstance(key, str):
            raise QueryError('filter: attribute needs a string "key"')
    if op != 'exists':
        allowed_keys.add('value')
        if 'value' not in x:
            raise QueryError(f'filter: {field} {op} needs a "value"')
    extra = set(x) - allowed_keys
    if extra:
        raise QueryError(f'filter: unexpected keys {sorted(extra)} in {field} {op}')

    if op == 'exists':
        value = None
    elif op == 'in':
        values = x['value']
        if not isinstance(values, list) or not values:
            raise QueryError(f'filter: {field} in needs a non-empty list')
        if len(values) > limits.max_in_values:
            raise QueryError(f'filter: at most {limits.max_in_values} values in an "in" list')
        value = tuple(parse_value(v) for v in values)
    else:
        value = parse_value(x['value'])
    return Leaf(field, op, value, key)


def leaves(node: Optional[FilterNode]) -> List[Leaf]:
    if node is None:
        return []
    if isinstance(node, Leaf):
        return [node]
    return [leaf for c in node.children for leaf in leaves(c)]


def job_id_bounds(node: Optional[FilterNode]) -> Tuple[int, int]:
    """The job id range a filter allows, from its top-level `and` `job_id` leaves; `or`s don't narrow it."""
    lo, hi = 0, MAX_JOB_ID
    if node is None or isinstance(node, Or):
        return lo, hi
    top = [node] if isinstance(node, Leaf) else [c for c in node.children if isinstance(c, Leaf)]
    for leaf in top:
        if leaf.field != 'job_id':
            continue
        if leaf.op == 'in':
            vs = cast(Tuple[int, ...], leaf.value)
            lo, hi = max(lo, min(vs)), min(hi, max(vs))
            continue
        v = cast(int, leaf.value)
        if leaf.op == '=':
            lo, hi = max(lo, v), min(hi, v)
        elif leaf.op == '<':
            hi = min(hi, v - 1)
        elif leaf.op == '<=':
            hi = min(hi, v)
        elif leaf.op == '>':
            lo = max(lo, v + 1)
        elif leaf.op == '>=':
            lo = max(lo, v)
    return lo, hi


def narrow_range(
    range_min: Optional[int], range_max: Optional[int], bounds: Tuple[int, int]
) -> Tuple[Optional[int], Optional[int]]:
    if range_min is None or range_max is None:
        return None, None
    lo, hi = max(range_min, bounds[0]), min(range_max, bounds[1])
    if lo > hi:
        return None, None
    return lo, hi


# Pagination


def plan_window(
    range_min: Optional[int],
    range_max: Optional[int],
    direction: str,
    start: Optional[int],
    max_scan_size: int,
) -> Union[Window, PageEnd]:
    """The window to scan, or the page's outcome if there's nothing to scan."""
    if range_min is None or range_max is None:
        return PageEnd(BOUNDARY, None, None)
    if direction == FORWARD:
        if start is None:
            start = range_min
        if start > range_max:
            return PageEnd(BOUNDARY, None, PageLink(range_max, BACKWARD))
        end = start + max_scan_size - 1
        if end < range_min:
            return PageEnd(SCAN_SIZE, PageLink(range_min, FORWARD), None)
        return Window(direction, start, max(start, range_min), min(end, range_max), range_min, range_max)
    if start is None:
        start = range_max
    if start < range_min:
        return PageEnd(BOUNDARY, PageLink(range_min, FORWARD), None)
    end = start - max_scan_size + 1
    if end > range_max:
        return PageEnd(SCAN_SIZE, None, PageLink(range_max, BACKWARD))
    return Window(direction, start, max(end, range_min), min(start, range_max), range_min, range_max)


def end_page(window: Window, page_ids: Sequence[int], limit: int, lookahead_found: bool, n_kept: int) -> PageEnd:
    """Why the page ended, and its links.

    `page_ids` are the first `limit` matches in scan order, `lookahead_found` whether the window held one
    more, and `n_kept` how many of `page_ids` the parent-edge budget kept.
    """
    forward = window.direction == FORWARD
    step = 1 if forward else -1

    if forward:
        edge_link = PageLink(window.hi + 1, FORWARD)
        other = PageLink(window.start - 1, BACKWARD) if window.start > window.range_min else None
    else:
        edge_link = PageLink(window.lo - 1, BACKWARD)
        other = PageLink(window.start + 1, FORWARD) if window.start < window.range_max else None

    reason: str
    link: Optional[PageLink]
    if n_kept < len(page_ids):
        reason, link = PARENT_EDGES, PageLink(page_ids[n_kept - 1] + step, window.direction)
    elif len(page_ids) == limit and lookahead_found:
        reason, link = LIMIT, PageLink(page_ids[-1] + step, window.direction)
    elif window.reaches_boundary:
        reason, link = BOUNDARY, None
    elif len(page_ids) == limit:
        reason, link = LIMIT, edge_link
    else:
        reason, link = SCAN_SIZE, edge_link

    if forward:
        return PageEnd(reason, link, other)
    return PageEnd(reason, other, link)


@dataclass(frozen=True)
class ParentEdges:
    n_kept: int
    parent_ids: Dict[int, List[int]]
    first_truncated: bool


def cut_parent_edges(page_ids: Sequence[int], edge_rows: Sequence[Tuple[int, int]], max_edges: int) -> ParentEdges:
    """Apply the parent-edge budget.

    `page_ids` are in scan order; `edge_rows` are (job_id, parent_id) in the same job order, at most
    `max_edges + 1` of them. Jobs before the one owning row `max_edges + 1` are kept; if that's the
    first job, it's kept alone with its first `max_edges` parents.
    """
    if len(edge_rows) <= max_edges:
        n_kept, kept_rows, first_truncated = len(page_ids), edge_rows, False
    else:
        cut_index = page_ids.index(edge_rows[max_edges][0])
        if cut_index == 0:
            n_kept, kept_rows, first_truncated = 1, edge_rows[:max_edges], True
        else:
            n_kept, first_truncated = cut_index, False
            kept = set(page_ids[:n_kept])
            kept_rows = [r for r in edge_rows if r[0] in kept]

    parent_ids: Dict[int, List[int]] = {job_id: [] for job_id in page_ids[:n_kept]}
    for job_id, parent_id in kept_rows:
        parent_ids[job_id].append(parent_id)
    for ps in parent_ids.values():
        ps.sort()
    return ParentEdges(n_kept, parent_ids, first_truncated)
