# pyright: strict
# gear's Transaction and BatchFormatVersion aren't fully typed; Unknown values from them are still reported where used.
# pyright: reportUnknownMemberType=false
# Statement.sql checks its argument's type at runtime too.
# pyright: reportUnnecessaryIsInstance=false
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

import pymysql
from typing_extensions import LiteralString

from gear import Database, Transaction

from ...batch_format_version import BatchFormatVersion
from ...exceptions import QueryError
from .job_list import (
    BACKWARD,
    And,
    FilterNode,
    JobListLimits,
    JobListParams,
    Leaf,
    LeafValue,
    Or,
    PageEnd,
    cut_parent_edges,
    end_page,
    job_id_bounds,
    leaves,
    narrow_range,
    plan_window,
)

MYSQL_QUERY_TIMEOUT = 3024

TERMINAL_STATES: Tuple[LiteralString, ...] = ('Cancelled', 'Error', 'Failed', 'Success')


class JobListTimeout(Exception):
    """The request's time budget ran out, in MySQL (error 3024) or before the next statement."""


class JobGroupNotFound(Exception):
    pass


@dataclass(frozen=True)
class Statement:
    """One SQL statement: everything after its first ``SELECT``, its args, and its table aliases.

    User input only ever goes in ``args``; ``body`` is built from fixed text, which ``LiteralString`` has pyright check.
    """

    name: str
    body: LiteralString
    args: Tuple[object, ...]
    aliases: Mapping[str, str] = field(default_factory=dict[str, str])

    def sql(self, time_limit_ms: int) -> str:
        # MySQL can't take a placeholder in a hint, and 0 would mean no limit.
        if not isinstance(time_limit_ms, int) or isinstance(time_limit_ms, bool) or time_limit_ms < 1:
            raise ValueError(f'bad time limit {time_limit_ms!r}')
        return f'SELECT /*+ MAX_EXECUTION_TIME({cast(LiteralString, str(time_limit_ms))}) */ {self.body}'


def _placeholders(n: int) -> LiteralString:
    placeholder: List[LiteralString] = ['%s']
    return ', '.join(placeholder * n)


def _exists(table: LiteralString, alias: LiteralString, where: LiteralString) -> LiteralString:
    # The one place a subquery's SELECT is written, so it's the one place a SQL-string scanner needs reviewing.
    return f'EXISTS (SELECT 1 FROM {table} AS {alias}\n  WHERE {where})'


def _same_job(alias: LiteralString) -> LiteralString:
    return f'{alias}.batch_id = jobs.batch_id AND {alias}.job_id = jobs.job_id'


def escape_like(s: str) -> str:
    return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


# Metadata


def groups_statement(batch_id: int, job_group_ids: Sequence[int]) -> Statement:
    return Statement(
        'job_list_groups',
        f"""
batches.format_version, batches.n_jobs AS batch_n_jobs, job_groups.job_group_id, job_groups.n_jobs
FROM job_groups
INNER JOIN batches ON batches.id = job_groups.batch_id
LEFT JOIN batch_updates
  ON batch_updates.batch_id = job_groups.batch_id AND batch_updates.update_id = job_groups.update_id
WHERE job_groups.batch_id = %s
  AND job_groups.job_group_id IN ({_placeholders(len(job_group_ids))})
  AND NOT batches.deleted
  AND (batch_updates.committed OR job_groups.job_group_id = 0)
""",
        (batch_id, *job_group_ids),
    )


def batch_range_statement(batch_id: int) -> Statement:
    # Updates can add only job groups; those have no ids to scan.
    return Statement(
        'job_list_batch_range',
        """
MIN(IF(committed, start_job_id, NULL)) AS min_job_id,
MAX(IF(committed, start_job_id + n_jobs - 1, NULL)) AS max_job_id,
MIN(IF(committed, NULL, start_job_id)) AS min_pending_job_id
FROM batch_updates
WHERE batch_id = %s AND n_jobs > 0
""",
        (batch_id,),
    )


def direct_group_range_statement(batch_id: int, job_group_ids: Sequence[int]) -> Statement:
    # One lookup per group: MySQL resolves each MIN/MAX from the ends of the index, which it can't do for an IN
    # list. The hint guarantees that plan; another index also starts (batch_id, job_group_id).
    one: LiteralString = """
MIN(job_id) AS min_job_id, MAX(job_id) AS max_job_id
FROM jobs FORCE INDEX (jobs_batch_id_job_group_id)
WHERE batch_id = %s AND job_group_id = %s
"""
    args: List[object] = []
    for job_group_id in job_group_ids:
        args.extend((batch_id, job_group_id))
    ones: List[LiteralString] = [one] * len(job_group_ids)
    return Statement('job_list_direct_group_range', ' UNION ALL SELECT '.join(ones), tuple(args))


def staged_pending_jobs_statement(batch_id: int, job_group_ids: Sequence[int]) -> Statement:
    # Uncommitted updates' jobs are already rows in `jobs`, so the recursive range lookup would read them.
    return Statement(
        'job_list_staged_pending_jobs',
        f"""
STRAIGHT_JOIN COALESCE(SUM(staging.n_jobs), 0) AS n_jobs
FROM batch_updates AS upd
INNER JOIN job_groups_inst_coll_staging AS staging
  ON staging.batch_id = upd.batch_id AND staging.update_id = upd.update_id
WHERE upd.batch_id = %s
  AND NOT upd.committed
  AND staging.job_group_id IN ({_placeholders(len(job_group_ids))})
""",
        (batch_id, *job_group_ids),
        {'upd': 'batch_updates', 'staging': 'job_groups_inst_coll_staging'},
    )


def subgroups_statement(batch_id: int, job_group_ids: Sequence[int], limit: int) -> Statement:
    return Statement(
        'job_list_subgroups',
        f"""
job_group_id
FROM job_group_self_and_ancestors
WHERE batch_id = %s AND ancestor_id IN ({_placeholders(len(job_group_ids))}) AND level > 0
LIMIT %s
""",
        (batch_id, *job_group_ids, limit),
    )


def recursive_group_range_statement(batch_id: int, job_group_ids: Sequence[int]) -> Statement:
    # A plain join. A per-sub-group LATERAL was measured reading the whole batch for each sub-group.
    return Statement(
        'job_list_recursive_group_range',
        f"""
STRAIGHT_JOIN MIN(j.job_id) AS min_job_id, MAX(j.job_id) AS max_job_id
FROM job_group_self_and_ancestors AS anc
INNER JOIN jobs AS j ON j.batch_id = anc.batch_id AND j.job_group_id = anc.job_group_id
WHERE anc.batch_id = %s AND anc.ancestor_id IN ({_placeholders(len(job_group_ids))})
""",
        (batch_id, *job_group_ids),
        {'anc': 'job_group_self_and_ancestors', 'j': 'jobs'},
    )


# The jobs query

_COMMITTED_JOIN: LiteralString = """
INNER JOIN batch_updates
  ON batch_updates.batch_id = jobs.batch_id AND batch_updates.update_id = jobs.update_id"""

_FRAGMENTS: Dict[str, LiteralString] = {
    'name': """
LEFT JOIN job_attributes AS name_attr
  ON name_attr.batch_id = jobs.batch_id AND name_attr.job_id = jobs.job_id AND name_attr.`key` = 'name'""",
    # latest_attempt_start/_end feed end_time and latest_attempt_duration; both are NULL when jobs.attempt_id is
    # (waiting for a retry, or preempted then cancelled).
    'attempt_summary': """
LEFT JOIN LATERAL (
  SELECT MIN(att.start_time) AS start_time,
         MAX(CASE WHEN att.attempt_id = jobs.attempt_id THEN att.start_time END) AS latest_attempt_start,
         MAX(CASE WHEN att.attempt_id = jobs.attempt_id THEN att.end_time END) AS latest_attempt_end
  FROM attempts AS att
  WHERE att.batch_id = jobs.batch_id AND att.job_id = jobs.job_id
) AS attempt_summary ON TRUE""",
    # NULL, not 0, for a job with no usage rows, so cost leaves don't match jobs that never ran.
    'cost': """
LEFT JOIN LATERAL (
  SELECT SUM(job_usage.`usage` * cost_resources.rate) AS cost
  FROM aggregated_job_resources_v3 AS job_usage
  INNER JOIN resources AS cost_resources ON cost_resources.resource_id = job_usage.resource_id
  WHERE job_usage.batch_id = jobs.batch_id AND job_usage.job_id = jobs.job_id
) AS cost_t ON TRUE""",
}

_FRAGMENT_ALIASES = {
    'name': {'name_attr': 'job_attributes'},
    'attempt_summary': {'att': 'attempts'},
    'cost': {'job_usage': 'aggregated_job_resources_v3', 'cost_resources': 'resources'},
}

_TERMINAL_SQL: LiteralString = 'jobs.state IN (' + ', '.join("'" + s + "'" for s in TERMINAL_STATES) + ')'
_END_TIME_SQL: LiteralString = f'IF({_TERMINAL_SQL}, attempt_summary.latest_attempt_end, NULL)'
_LATEST_ATTEMPT_DURATION_SQL: LiteralString = (
    '(attempt_summary.latest_attempt_end - attempt_summary.latest_attempt_start)'
)
_EXIT_CODE_JSON: LiteralString = "JSON_EXTRACT(jobs.status, '$[0]')"

# field -> (SQL expression, fragment it needs)
_COLUMNS: Dict[str, Tuple[LiteralString, Optional[str]]] = {
    'job_id': ('jobs.job_id', None),
    'state': ('jobs.state', None),
    'instance_collection': ('jobs.inst_coll', None),
    'name': ('name_attr.value', 'name'),
    'start_time': ('attempt_summary.start_time', 'attempt_summary'),
    'end_time': (_END_TIME_SQL, 'attempt_summary'),
    'duration': (f'({_END_TIME_SQL} - attempt_summary.start_time)', 'attempt_summary'),
    'latest_attempt_duration': (_LATEST_ATTEMPT_DURATION_SQL, 'attempt_summary'),
    'cost': ('cost_t.cost', 'cost'),
    # A JSON null isn't a SQL NULL: unguarded, an unknown exit code would cast to 0.
    'exit_code': (f'CAST({_EXIT_CODE_JSON} AS SIGNED)', None),
}

_INCLUDE_FRAGMENTS = {
    'start_time': 'attempt_summary',
    'end_time': 'attempt_summary',
    'latest_attempt_duration': 'attempt_summary',
    'cost': 'cost',
}

_SQL_OPS: Dict[str, LiteralString] = {'=': '=', '!=': '<>', '<': '<', '<=': '<=', '>': '>', '>=': '>='}


class _FilterCompiler:
    def __init__(self):
        self.args: List[object] = []
        self.fragments: List[str] = []
        self.aliases: Dict[str, str] = {}
        self._n_aliases = 0

    def _alias(self, prefix: LiteralString, table: str) -> LiteralString:
        alias = f'{prefix}_{cast(LiteralString, str(self._n_aliases))}'
        self._n_aliases += 1
        self.aliases[alias] = table
        return alias

    def _need(self, fragment: Optional[str]):
        if fragment is not None and fragment not in self.fragments:
            self.fragments.append(fragment)

    def _compare(self, expr: LiteralString, op: str, value: LeafValue) -> LiteralString:
        if op == 'in':
            assert isinstance(value, tuple)
            self.args.extend(value)
            return f'{expr} IN ({_placeholders(len(value))})'
        if op in ('contains', 'not_contains'):
            assert isinstance(value, str)
            self.args.append(f'%{escape_like(value)}%')
            return f'{expr} {"LIKE" if op == "contains" else "NOT LIKE"} %s'
        self.args.append(value)
        return f'{expr} {_SQL_OPS[op]} %s'

    def node(self, node: FilterNode) -> LiteralString:
        if isinstance(node, And):
            return '(' + ' AND '.join(self.node(c) for c in node.children) + ')'
        if isinstance(node, Or):
            return '(' + ' OR '.join(self.node(c) for c in node.children) + ')'
        return self.leaf(node)

    def leaf(self, leaf: Leaf) -> LiteralString:
        if leaf.field == 'attribute':
            a = self._alias('attr', 'job_attributes')
            self.args.append(leaf.key)
            cond = '' if leaf.op == 'exists' else ' AND ' + self._compare(f'{a}.value', leaf.op, leaf.value)
            return _exists('job_attributes', a, f'{_same_job(a)} AND {a}.`key` = %s{cond}')
        if leaf.field == 'text':
            a = self._alias('text_attr', 'job_attributes')
            t = self._alias('text_att', 'attempts')
            key_cond = self._compare(f'{a}.`key`', leaf.op, leaf.value)
            value_cond = self._compare(f'{a}.value', leaf.op, leaf.value)
            instance_cond = self._compare(f'{t}.instance_name', leaf.op, leaf.value)
            attr_exists = _exists('job_attributes', a, f'{_same_job(a)} AND ({key_cond} OR {value_cond})')
            attempt_exists = _exists('attempts', t, f'{_same_job(t)} AND {instance_cond}')
            return f'({attr_exists}\n OR {attempt_exists})'
        if leaf.field == 'instance':
            t = self._alias('inst_att', 'attempts')
            cond = self._compare(f'{t}.instance_name', leaf.op, leaf.value)
            return _exists('attempts', t, f'{_same_job(t)} AND {cond}')
        expr, fragment = _COLUMNS[leaf.field]
        self._need(fragment)
        cond = self._compare(expr, leaf.op, leaf.value)
        if leaf.field == 'exit_code':
            return f"(JSON_TYPE({_EXIT_CODE_JSON}) = 'INTEGER' AND {cond})"
        return f'({cond})'


@dataclass(frozen=True)
class GroupFilter:
    job_group_ids: Tuple[int, ...]
    recursive: bool


def _jobs_from_where(
    batch_id: int,
    lo: int,
    hi: int,
    group_filter: Optional[GroupFilter],
    filter_: Optional[FilterNode],
    display_fragments: Sequence[str],
) -> Tuple[LiteralString, List[object], Dict[str, str]]:
    """FROM through WHERE, shared by the jobs query and the count."""
    conditions: List[LiteralString] = ['jobs.batch_id = %s', 'jobs.job_id BETWEEN %s AND %s', 'batch_updates.committed']
    args: List[object] = [batch_id, lo, hi]
    aliases: Dict[str, str] = {}

    if group_filter is not None:
        ids = group_filter.job_group_ids
        if group_filter.recursive:
            # A semi-join, not a JOIN: with overlapping groups a JOIN would return a job more than once.
            conditions.append(
                _exists(
                    'job_group_self_and_ancestors',
                    'grp',
                    'grp.batch_id = jobs.batch_id AND grp.job_group_id = jobs.job_group_id'
                    f'\n    AND grp.ancestor_id IN ({_placeholders(len(ids))})',
                )
            )
            aliases['grp'] = 'job_group_self_and_ancestors'
        else:
            conditions.append(f'jobs.job_group_id IN ({_placeholders(len(ids))})')
        args.extend(ids)

    compiler = _FilterCompiler()
    if filter_ is not None:
        conditions.append(compiler.node(filter_))
        args.extend(compiler.args)
        aliases.update(compiler.aliases)

    # STRAIGHT_JOIN keeps this order: joins the filter uses first, so display-only ones run only for matches.
    fragments = list(dict.fromkeys([*compiler.fragments, *display_fragments]))
    for f in fragments:
        aliases.update(_FRAGMENT_ALIASES[f])

    from_where = (
        'FROM jobs'
        + _COMMITTED_JOIN
        + ''.join(_FRAGMENTS[f] for f in fragments)
        + '\nWHERE '
        + '\n  AND '.join(conditions)
    )
    return from_where, args, aliases


def jobs_statement(
    batch_id: int,
    lo: int,
    hi: int,
    direction: str,
    limit: int,
    group_filter: Optional[GroupFilter],
    filter_: Optional[FilterNode],
    include: Sequence[str],
) -> Statement:
    display = ['name'] + [_INCLUDE_FRAGMENTS[i] for i in include if i in _INCLUDE_FRAGMENTS]
    from_where, args, aliases = _jobs_from_where(batch_id, lo, hi, group_filter, filter_, display)

    columns: List[LiteralString] = ['jobs.job_id', 'jobs.job_group_id', 'jobs.state', 'name_attr.value AS name']
    if 'attempt_summary' in display:
        columns.extend([
            'attempt_summary.start_time AS start_time',
            f'{_END_TIME_SQL} AS end_time',
            f'{_LATEST_ATTEMPT_DURATION_SQL} AS latest_attempt_duration',
        ])
    if 'exit_code' in include:
        columns.append('jobs.status')
    if 'cost' in include:
        columns.append('cost_t.cost')

    order = 'DESC' if direction == BACKWARD else 'ASC'
    body = f"""STRAIGHT_JOIN {', '.join(columns)}
{from_where}
ORDER BY jobs.job_id {order}
LIMIT %s
"""
    return Statement('job_list_jobs', body, (*args, limit), aliases)


def count_statement(
    batch_id: int, lo: int, hi: int, group_filter: Optional[GroupFilter], filter_: Optional[FilterNode]
) -> Statement:
    from_where, args, aliases = _jobs_from_where(batch_id, lo, hi, group_filter, filter_, [])
    return Statement('job_list_count', f'STRAIGHT_JOIN COUNT(*) AS n\n{from_where}\n', tuple(args), aliases)


# Follow-ups, once the page's ids are known


def attempts_statement(batch_id: int, job_ids: Sequence[int]) -> Statement:
    return Statement(
        'job_list_attempts',
        f"""
job_id, attempt_id, instance_name, start_time, end_time, reason
FROM attempts
WHERE batch_id = %s AND job_id IN ({_placeholders(len(job_ids))})
""",
        (batch_id, *job_ids),
    )


def cost_per_hour_statement(batch_id: int, job_ids: Sequence[int]) -> Statement:
    # Reads attempt_resources, which the Cloud SQL cleanup may delete or compact for completed batches: null rates
    # for old batches are expected. If the table goes away, move the rate to durable storage rather than treat
    # this as a bug.
    return Statement(
        'job_list_cost_per_hour',
        f"""
ar.job_id, ar.attempt_id, SUM(ar.quantity * r.rate) * 3600000 AS cost_per_hour
FROM attempt_resources AS ar
INNER JOIN resources AS r ON r.resource_id = COALESCE(ar.deduped_resource_id, ar.resource_id)
WHERE ar.batch_id = %s AND ar.job_id IN ({_placeholders(len(job_ids))})
GROUP BY ar.job_id, ar.attempt_id
""",
        (batch_id, *job_ids),
        {'ar': 'attempt_resources', 'r': 'resources'},
    )


def parent_edges_statement(batch_id: int, job_ids: Sequence[int], direction: str, limit: int) -> Statement:
    order = 'DESC' if direction == BACKWARD else 'ASC'
    return Statement(
        'job_list_parent_edges',
        f"""
job_id, parent_id
FROM job_parents
WHERE batch_id = %s AND job_id IN ({_placeholders(len(job_ids))})
ORDER BY job_id {order}, parent_id {order}
LIMIT %s
""",
        (batch_id, *job_ids, limit),
    )


# Running a request


# Rows are object, not Any, so a value read back can't reach SQL text without pyright noticing.
Row = Mapping[str, object]


def _opt_int(row: Row, key: str) -> Optional[int]:
    v = row[key]
    # SUM comes back as a Decimal.
    if isinstance(v, Decimal) and v == v.to_integral_value():
        return int(v)
    assert v is None or isinstance(v, int), (key, v)
    return v


def _int(row: Row, key: str) -> int:
    v = _opt_int(row, key)
    assert v is not None, key
    return v


def _opt_float(row: Row, key: str) -> Optional[float]:
    v = row[key]
    if v is None:
        return None
    assert isinstance(v, (int, float, Decimal)), (key, v)
    return float(v)


def _opt_str(row: Row, key: str) -> Optional[str]:
    v = row[key]
    assert v is None or isinstance(v, str), (key, v)
    return v


class _Runner:
    """Runs a request's statements in one transaction against one deadline."""

    def __init__(self, tx: Transaction, time_limit_ms: int, clock: Callable[[], float]):
        self._tx = tx
        self._clock = clock
        self._end = clock() + time_limit_ms / 1000

    async def all(self, stmt: Statement) -> List[Row]:
        remaining_ms = int((self._end - self._clock()) * 1000)
        if remaining_ms < 1:
            raise JobListTimeout()
        try:
            # The Transaction's own methods: Database's retry transient errors with no limit, so a lost
            # connection could overrun the deadline without bound.
            return [r async for r in self._tx.execute_and_fetchall(stmt.sql(remaining_ms), stmt.args, stmt.name)]
        except pymysql.err.OperationalError as e:
            if e.args and e.args[0] == MYSQL_QUERY_TIMEOUT:
                raise JobListTimeout() from e
            raise

    async def one(self, stmt: Statement) -> Row:
        rows = await self.all(stmt)
        assert len(rows) == 1, (stmt.name, rows)
        return rows[0]


def _hull(rows: Sequence[Row]) -> Tuple[Optional[int], Optional[int]]:
    mins = [v for r in rows if (v := _opt_int(r, 'min_job_id')) is not None]
    maxes = [v for r in rows if (v := _opt_int(r, 'max_job_id')) is not None]
    return (min(mins) if mins else None, max(maxes) if maxes else None)


async def _recursive_range(
    run: _Runner,
    batch_id: int,
    job_group_ids: Sequence[int],
    groups: Sequence[Row],
    limits: JobListLimits,
) -> Optional[Tuple[Optional[int], Optional[int]]]:
    """The groups' range including sub-groups, or None if looking it up could be too expensive."""
    committed = sum(_int(g, 'n_jobs') for g in groups)
    if committed > limits.group_range_lookup_max_jobs:
        return None
    staged = await run.one(staged_pending_jobs_statement(batch_id, job_group_ids))
    if committed + _int(staged, 'n_jobs') > limits.group_range_lookup_max_jobs:
        return None
    # The jobs gate doesn't bound this: sub-groups can be empty, and most of the lookup's cost is per sub-group.
    subgroups = await run.all(subgroups_statement(batch_id, job_group_ids, limits.group_range_lookup_max_subgroups + 1))
    if len(subgroups) > limits.group_range_lookup_max_subgroups:
        return None
    return _hull([await run.one(recursive_group_range_statement(batch_id, job_group_ids))])


async def _total_jobs(
    run: _Runner,
    batch_id: int,
    params: JobListParams,
    groups: Sequence[Row],
    group_filter: Optional[GroupFilter],
    range_min: Optional[int],
    range_max: Optional[int],
    limits: JobListLimits,
    root_recursive: bool,
) -> Optional[int]:
    """Exact, or None when it could be too big to count cheaply. Runs last, on what's left of the budget."""
    if range_min is None or range_max is None:
        return 0
    if range_max - range_min + 1 <= limits.total_jobs_count_max:
        try:
            row = await run.one(count_statement(batch_id, range_min, range_max, group_filter, params.filter))
        except JobListTimeout:
            return None
        return _int(row, 'n')
    if params.filter is None and params.recursive and len(params.job_group_ids) == 1:
        return _int(groups[0], 'batch_n_jobs' if root_recursive else 'n_jobs')
    return None


def _job_json(
    row: Row,
    include: Set[str],
    format_version: BatchFormatVersion,
    attempts: Mapping[int, List[Dict[str, object]]],
    parent_ids: Mapping[int, List[int]],
    first_truncated: bool,
) -> Dict[str, Any]:
    job_id = _int(row, 'job_id')
    return {
        'job_id': job_id,
        'job_group_id': row['job_group_id'],
        'name': row['name'],
        'state': row['state'],
        'start_time': row['start_time'] if 'start_time' in include else None,
        'end_time': row['end_time'] if 'end_time' in include else None,
        'latest_attempt_duration': row['latest_attempt_duration'] if 'latest_attempt_duration' in include else None,
        'exit_code': _exit_code(format_version, _opt_str(row, 'status')) if 'exit_code' in include else None,
        'cost': _opt_float(row, 'cost') if 'cost' in include else None,
        'attempts': attempts.get(job_id, []) if 'attempts' in include else None,
        'parent_ids': parent_ids.get(job_id, []) if 'parent_ids' in include else None,
        # Only a lone over-budget job is ever truncated, so it's the page's only row.
        'parent_ids_truncated': first_truncated if 'parent_ids' in include else None,
    }


def _exit_code(format_version: BatchFormatVersion, status: Optional[str]) -> Optional[int]:
    if not status:
        return None
    exit_code, _ = format_version.get_status_exit_code_duration(json.loads(status))
    return exit_code


async def get_job_list(
    db: Database,
    batch_id: int,
    params: JobListParams,
    limits: JobListLimits,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    include = set(params.include)
    filter_leaves = leaves(params.filter)

    async with db.start(read_only=True) as tx:
        run = _Runner(tx, limits.query_time_limit_ms, clock)

        groups = await run.all(groups_statement(batch_id, params.job_group_ids))
        if len(groups) != len(params.job_group_ids):
            raise JobGroupNotFound()
        format_version = BatchFormatVersion(_int(groups[0], 'format_version'))
        if format_version.has_full_status_in_db() and any(leaf.field == 'exit_code' for leaf in filter_leaves):
            raise QueryError('filter: exit_code is not supported for this batch')

        # Metadata before rows: an update committing mid-request then only makes the response more conservative.
        batch_range = await run.one(batch_range_statement(batch_id))
        batch_min, batch_max = _opt_int(batch_range, 'min_job_id'), _opt_int(batch_range, 'max_job_id')
        min_pending_job_id = _opt_int(batch_range, 'min_pending_job_id')
        if min_pending_job_id is not None:
            stable_below_job_id = min_pending_job_id
        elif batch_max is not None:
            stable_below_job_id = batch_max + 1
        else:
            stable_below_job_id = 1

        root_recursive = params.recursive and 0 in params.job_group_ids
        group_filter = None if root_recursive else GroupFilter(params.job_group_ids, params.recursive)
        if root_recursive:
            range_min, range_max = batch_min, batch_max
        elif not params.recursive:
            range_min, range_max = _hull(await run.all(direct_group_range_statement(batch_id, params.job_group_ids)))
        else:
            group_range = await _recursive_range(run, batch_id, params.job_group_ids, groups, limits)
            range_min, range_max = group_range if group_range is not None else (batch_min, batch_max)

        range_min, range_max = narrow_range(range_min, range_max, job_id_bounds(params.filter))

        rows: List[Row] = []
        page_end: PageEnd
        parent_ids: Dict[int, List[int]] = {}
        first_truncated = False
        if params.limit == 0:
            page_end = PageEnd(None, None, None)
        else:
            plan = plan_window(
                range_min, range_max, params.scan_direction, params.scan_start_job_id, params.max_scan_size
            )
            if isinstance(plan, PageEnd):
                page_end = plan
            else:
                found = await run.all(
                    jobs_statement(
                        batch_id,
                        plan.lo,
                        plan.hi,
                        params.scan_direction,
                        params.limit + 1,
                        group_filter,
                        params.filter,
                        params.include,
                    )
                )
                rows = found[: params.limit]
                page_ids = [_int(r, 'job_id') for r in rows]
                n_kept = len(rows)
                if 'parent_ids' in include and rows:
                    edge_rows = await run.all(
                        parent_edges_statement(
                            batch_id, page_ids, params.scan_direction, limits.parent_edge_page_max + 1
                        )
                    )
                    edges = cut_parent_edges(
                        page_ids,
                        [(_int(r, 'job_id'), _int(r, 'parent_id')) for r in edge_rows],
                        limits.parent_edge_page_max,
                    )
                    n_kept, parent_ids, first_truncated = edges.n_kept, edges.parent_ids, edges.first_truncated
                page_end = end_page(plan, page_ids, params.limit, len(found) > params.limit, n_kept)
                rows = rows[:n_kept]

        rows.sort(key=lambda r: _int(r, 'job_id'))
        job_ids = [_int(r, 'job_id') for r in rows]

        attempts: Dict[int, List[Dict[str, object]]] = {}
        if 'attempts' in include and job_ids:
            for a in await run.all(attempts_statement(batch_id, job_ids)):
                attempts.setdefault(_int(a, 'job_id'), []).append({
                    'attempt_id': a['attempt_id'],
                    'instance_name': a['instance_name'],
                    'start_time': a['start_time'],
                    'end_time': a['end_time'],
                    'reason': a['reason'],
                    'cost_per_hour': None,
                })
            for job_attempts in attempts.values():
                job_attempts.sort(key=lambda a: (a['start_time'] is None, a['start_time'], a['attempt_id']))
            if 'attempts.cost_per_hour' in include:
                rates = {
                    (_int(r, 'job_id'), r['attempt_id']): rate
                    for r in await run.all(cost_per_hour_statement(batch_id, job_ids))
                    if (rate := _opt_float(r, 'cost_per_hour')) is not None
                }
                for job_id, job_attempts in attempts.items():
                    for a in job_attempts:
                        a['cost_per_hour'] = rates.get((job_id, a['attempt_id']))

        total_jobs = None
        if 'total_jobs' in include:
            total_jobs = await _total_jobs(
                run, batch_id, params, groups, group_filter, range_min, range_max, limits, root_recursive
            )

    return {
        'jobs': [_job_json(r, include, format_version, attempts, parent_ids, first_truncated) for r in rows],
        'include': list(params.include),
        'pagination': {
            'first_job_id': job_ids[0] if job_ids else None,
            'last_job_id': job_ids[-1] if job_ids else None,
            'next_page': page_end.next_page.to_json() if page_end.next_page else None,
            'previous_page': page_end.previous_page.to_json() if page_end.previous_page else None,
            'page_end_reason': page_end.reason,
            'scan_range': {'min': range_min, 'max': range_max, 'stable_below_job_id': stable_below_job_id},
            'total_jobs': total_jobs,
        },
    }
