"""Validators for SQL queries' plans and row reads, run against a real MySQL."""

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from gear import Database

# Small, fixed tables, which assert_scoped lets a query scan in full.
SMALL_LOOKUP_TABLES = ('resources', 'inst_colls', 'regions', 'globals', 'feature_flags')

_LOOKUP_ACCESS_TYPES = ('system', 'const', 'eq_ref', 'ref', 'ref_or_null')

_SUBQUERY_KEYS = (
    'attached_subqueries',
    'select_list_subqueries',
    'having_subqueries',
    'order_by_subqueries',
    'group_by_subqueries',
    'update_value_subqueries',
)


async def _handler_reads(tx) -> int:
    rows = [r async for r in tx.execute_and_fetchall("SHOW SESSION STATUS LIKE 'Handler_read%'")]
    return sum(int(r['Value']) for r in rows)


async def count_row_reads(db: Database, sql: str, args: Optional[Sequence[Any]] = None) -> Tuple[List[dict], int]:
    """Run ``sql`` and return its rows and the number of rows the storage engine read for it.

    Reading the counters reads rows too, so that cost is measured and subtracted.
    """
    async with db.start(read_only=True) as tx:
        a = await _handler_reads(tx)
        b = await _handler_reads(tx)
        show_cost = b - a
        rows = [r async for r in tx.execute_and_fetchall(sql, args)]
        c = await _handler_reads(tx)
    return rows, c - b - show_cost


@dataclass
class TableAccess:
    table: str
    access_type: Optional[str]
    key: Optional[str]
    used_key_parts: List[str]
    ref: List[str]  # what each used key part was matched against: 'const', a column 'db.alias.col', or 'func'
    path: str
    derived: bool = False  # a materialized subquery; the tables it reads appear as accesses of their own


@dataclass
class Subquery:
    dependent: bool
    path: str


@dataclass
class Materialization:
    table: str
    dependent: bool  # built per outer row (e.g. a LATERAL), not once over its whole input
    path: str


@dataclass
class Plan:
    raw: Dict[str, Any]
    accesses: List[TableAccess] = field(default_factory=list)
    subqueries: List[Subquery] = field(default_factory=list)
    materialized: List['Materialization'] = field(default_factory=list)
    filesorts: List[str] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    base_tables: Set[str] = field(default_factory=set)
    batch_scope_columns: Dict[str, str] = field(default_factory=dict)  # table -> its batch id column
    # table or alias -> each range scan's ranges; only the tree format has them
    ranges: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def rewritten_sql(self) -> Optional[str]:
        """The statement as the optimizer rewrote it, with the hints it kept."""
        return next((w['Message'] for w in self.warnings if w['Code'] == 1003), None)

    def accesses_to(self, table: str) -> List[TableAccess]:
        return [a for a in self.accesses if a.table == table]


def _walk(node: Any, path: str, plan: Plan):
    if isinstance(node, list):
        for i, x in enumerate(node):
            _walk(x, f'{path}[{i}]', plan)
        return
    if not isinstance(node, dict):
        return

    materialized = node.get('materialized_from_subquery')
    if 'table_name' in node:
        plan.accesses.append(
            TableAccess(
                table=node['table_name'],
                access_type=node.get('access_type'),
                key=node.get('key'),
                used_key_parts=list(node.get('used_key_parts', [])),
                ref=list(node.get('ref', [])),
                path=path,
                derived=materialized is not None,
            )
        )
    if materialized is not None:
        plan.materialized.append(
            Materialization(node.get('table_name', '?'), bool(materialized.get('dependent')), path)
        )
    if node.get('using_filesort'):
        plan.filesorts.append(path)
    for key in _SUBQUERY_KEYS:
        for i, sq in enumerate(node.get(key, [])):
            plan.subqueries.append(Subquery(dependent=bool(sq.get('dependent')), path=f'{path}.{key}[{i}]'))

    for k, v in node.items():
        _walk(v, f'{path}.{k}', plan)


async def _batch_scope_columns(tx) -> Dict[str, str]:
    """Each per-batch table's batch id column, from the schema so new tables are covered. Foreign keys to
    ``batches.id`` find the tables whose column is named ``id``; ``batch_id`` columns catch any without one."""
    rows = tx.execute_and_fetchall(
        """
SELECT table_name AS t, column_name AS c FROM information_schema.key_column_usage
WHERE table_schema = DATABASE() AND referenced_table_name = 'batches' AND referenced_column_name = 'id'
UNION
SELECT table_name AS t, column_name AS c FROM information_schema.columns
WHERE table_schema = DATABASE() AND column_name = 'batch_id';
"""
    )
    columns = {'batches': 'id'}
    async for r in rows:
        assert columns.get(r['t'], r['c']) == r['c'], f"{r['t']} has two batch columns: {columns[r['t']]}, {r['c']}"
        columns[r['t']] = r['c']
    return columns


async def explain(db: Database, sql: str, args: Optional[Sequence[Any]] = None) -> Plan:
    """Explain ``sql``, with what the checks below need from the schema."""
    async with db.start(read_only=True) as tx:
        # EXPLAIN always leaves a note, which aiomysql raises as a Python warning; it's read with SHOW WARNINGS.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            row = await tx.execute_and_fetchone(f'EXPLAIN FORMAT=JSON {sql}', args)
        mysql_warnings = [w async for w in tx.execute_and_fetchall('SHOW WARNINGS')]
        # Only the tree format shows a range scan's ranges.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tree = await tx.execute_and_fetchone(f'EXPLAIN FORMAT=TREE {sql}', args)
        base_tables = {
            r['name']
            async for r in tx.execute_and_fetchall(
                'SELECT table_name AS name FROM information_schema.tables WHERE table_schema = DATABASE()'
            )
        }
        batch_scope_columns = await _batch_scope_columns(tx)
    raw = json.loads(row['EXPLAIN'])
    plan = Plan(raw=raw, warnings=mysql_warnings, base_tables=base_tables, batch_scope_columns=batch_scope_columns)
    _walk(raw, '$', plan)
    for line in tree['EXPLAIN'].splitlines():
        m = re.search(r'range scan on (\S+) using \S+ over (.*?)(?:\s+\(cost=.*)?$', line)
        if m:
            plan.ranges.setdefault(m.group(1), []).append(m.group(2))
    return plan


def _single_batch(ranges: str, column: str) -> bool:
    """``(batch_id = 7 AND 100 <= job_id)``, but not ``(1 <= batch_id <= 2)`` or ``(batch_id = 1) OR (batch_id = 2)``."""
    mentions = re.findall(rf'\b{column}\b', ranges)
    values = re.findall(rf'\b{column} = ([^\s)]+)', ranges)
    return len(values) == len(mentions) and len(set(values)) == 1


def assert_scoped(
    plan: Plan,
    *,
    aliases: Optional[Mapping[str, str]] = None,  # alias -> table, e.g. {'latest_attempt': 'attempts'}
    allow_full_scan: Sequence[str] = SMALL_LOOKUP_TABLES,
    allow_filesort: bool = False,
):
    """Every per-batch table access reads one batch, every subquery and derived table is dependent, nothing
    outside ``allow_full_scan`` is fully scanned, and nothing is filesorted unless ``allow_filesort`` (a filesort
    reads every candidate row before a ``LIMIT`` can stop it).

    An access reads one batch when its index's batch column is matched against a constant, fixed to one value
    in every range of a range scan, or matched against the batch column of an access already shown to read one
    batch. Using an index on the batch column isn't enough: ``batch_id BETWEEN 1 AND 2`` and
    ``jobs.batch_id = resources.resource_id`` both do.

    ``EXPLAIN`` names tables by alias, so pass the query's aliases; an unknown name fails rather than going
    unchecked. Derived tables are skipped, since the tables they read are checked as accesses of their own.
    """
    scope_columns = plan.batch_scope_columns
    problems = []
    candidates = []  # (access, display name, batch column)
    for a in plan.accesses:
        if a.derived or a.table.startswith('<'):
            continue
        table = (aliases or {}).get(a.table, a.table)
        name = table if table == a.table else f'{a.table} (alias of {table})'
        if table not in plan.base_tables:
            problems.append(f'{a.table}: not a table; pass its table in aliases at {a.path}')
            continue
        column = scope_columns.get(table)
        if a.access_type in ('ALL', 'index'):
            if column is not None or table not in allow_full_scan:
                problems.append(f'{name}: full scan (access_type={a.access_type}) at {a.path}')
        elif column is None:
            continue
        elif a.access_type == 'index_merge':
            # proven below from each merged index's ranges, which EXPLAIN FORMAT=JSON doesn't list as key parts
            candidates.append((a, name, column))
        elif column not in a.used_key_parts:
            problems.append(f'{name}: key {a.key} used_key_parts {a.used_key_parts} lacks {column} at {a.path}')
        else:
            candidates.append((a, name, column))

    # Repeat until nothing changes: a lookup is proven only once the access feeding it is.
    proven: Set[int] = set()  # ids of proven accesses
    unproven_reason: Dict[int, str] = {}

    def alias_proven(alias: str, col: str) -> bool:
        matches = [(a, c) for a, _, c in candidates if a.table == alias]
        return bool(matches) and all(id(a) in proven and c == col for a, c in matches)

    changed = True
    while changed:
        changed = False
        for a, _, column in candidates:
            if id(a) in proven:
                continue
            if a.access_type in ('range', 'index_merge'):
                scans = plan.ranges.get(a.table)
                if a.access_type == 'index_merge' and scans:
                    # the merged indexes' ranges must all fix the same batch, not one each
                    scans = [' OR '.join(scans)]
                bad = [r for r in scans or [] if not _single_batch(r, column)]
                if scans and not bad:
                    proven.add(id(a))
                    changed = True
                else:
                    unproven_reason[id(a)] = (
                        f'range {bad[0]} is not a single {column}'
                        if bad
                        else 'range scan with no ranges in the tree plan'
                    )
            elif a.access_type in _LOOKUP_ACCESS_TYPES:
                i = a.used_key_parts.index(column)
                ref = a.ref[i] if i < len(a.ref) else None
                parts = (ref or '').split('.')
                if ref == 'const' or (len(parts) == 3 and alias_proven(parts[1], parts[2])):
                    proven.add(id(a))
                    changed = True
                else:
                    unproven_reason[id(a)] = f'{column} is matched against {ref}, not one batch'
            else:
                unproven_reason[id(a)] = f'access type {a.access_type} cannot be proven to read one batch'
    problems.extend(f'{name}: {unproven_reason[id(a)]} at {a.path}' for a, name, _ in candidates if id(a) not in proven)

    problems.extend(f'non-dependent subquery at {s.path}' for s in plan.subqueries if not s.dependent)
    problems.extend(f'non-dependent materialized {m.table} at {m.path}' for m in plan.materialized if not m.dependent)
    if not allow_filesort:
        problems.extend(f'filesort at {p}' for p in plan.filesorts)
    assert not problems, '\n'.join(problems) + '\n' + json.dumps(plan.raw, indent=2)


def assert_hint_kept(plan: Plan, hint: str = 'MAX_EXECUTION_TIME'):
    """The optimizer kept ``hint``. An ignored or misplaced one (e.g. on a subquery) only produces a warning."""
    rewritten = plan.rewritten_sql
    assert rewritten is not None, plan.warnings
    assert f'/*+ {hint}(' in rewritten, rewritten
    bad = [w for w in plan.warnings if w['Level'] != 'Note']
    assert not bad, bad
