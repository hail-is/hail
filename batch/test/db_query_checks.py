"""Assertions that a query is scoped to one batch and bounded, independent of the data it happens to return.

Two complementary checks, both against a real MySQL:

- :func:`count_row_reads` counts the rows the storage engine actually read (``Handler_read%``), so a test can
  bound it by what the request asked for (window size, ``LIMIT``, group size) whatever plan MySQL picks.
  Seed a large noise batch next to the batch under test so an unscoped read shows up.
- :func:`explain` + :func:`assert_scoped` check the plan's structure: every access to a per-batch table uses
  an index on its batch column, subqueries are dependent rather than materialized, and nothing is filesorted.
  The per-batch tables are read from the schema (see :func:`_batch_scope_columns`), so new ones are covered.
  ``EXPLAIN`` names tables by their alias, so queries that alias a table pass ``aliases`` to
  :func:`assert_scoped`; an access it can't resolve to a real table fails rather than being skipped.
  :func:`assert_hint_kept` checks an optimizer hint survived (e.g. ``MAX_EXECUTION_TIME``).

Run ``db_seed.analyze_tables`` after seeding so the optimizer sees realistic row counts.
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from gear import Database

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
    """Run ``sql`` and return its rows and the number of handler row reads it made.

    Reads the session ``Handler_read%`` counters before and after, minus the cost of reading the counters
    themselves (``SHOW STATUS`` reads rows too), all on one connection.
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
    path: str


@dataclass
class Subquery:
    dependent: bool
    cacheable: bool
    path: str


@dataclass
class Plan:
    raw: Dict[str, Any]
    accesses: List[TableAccess] = field(default_factory=list)
    subqueries: List[Subquery] = field(default_factory=list)
    materialized: List[str] = field(default_factory=list)  # paths of materialized derived tables/subqueries
    filesorts: List[str] = field(default_factory=list)  # paths of ordering/grouping operations using filesort
    warnings: List[Dict[str, Any]] = field(default_factory=list)  # SHOW WARNINGS after the EXPLAIN
    base_tables: Set[str] = field(default_factory=set)  # the database's real table names
    # per-batch table -> the column holding its batch id (batch_id, or id for batches and a few counters)
    batch_scope_columns: Dict[str, str] = field(default_factory=dict)

    @property
    def rewritten_sql(self) -> Optional[str]:
        """The statement as the optimizer rewrote it (warning 1003), including the hints it kept."""
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

    if 'table_name' in node:
        plan.accesses.append(
            TableAccess(
                table=node['table_name'],
                access_type=node.get('access_type'),
                key=node.get('key'),
                used_key_parts=list(node.get('used_key_parts', [])),
                path=path,
            )
        )
    if 'materialized_from_subquery' in node:
        plan.materialized.append(path)
    if node.get('using_filesort'):
        plan.filesorts.append(path)
    for key in _SUBQUERY_KEYS:
        for i, sq in enumerate(node.get(key, [])):
            plan.subqueries.append(
                Subquery(
                    dependent=bool(sq.get('dependent')), cacheable=bool(sq.get('cacheable')), path=f'{path}.{key}[{i}]'
                )
            )

    for k, v in node.items():
        _walk(v, f'{path}.{k}', plan)


async def _batch_scope_columns(tx) -> Dict[str, str]:
    """Every per-batch table and the column holding its batch id: ``batches.id`` itself, every column with a
    foreign key to it (which covers tables whose batch column is named ``id``), and every ``batch_id`` column
    (in case a migration dropped a foreign key). Read from the schema so that new tables are covered."""
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
    """``EXPLAIN FORMAT=JSON`` the statement, and collect the warnings it leaves (on the same connection)."""
    async with db.start(read_only=True) as tx:
        # EXPLAIN always leaves a note (1003, the rewritten statement); aiomysql would re-raise it as a Python
        # warning, which pytest.ini turns into an error. It's read back below instead.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            row = await tx.execute_and_fetchone(f'EXPLAIN FORMAT=JSON {sql}', args)
        mysql_warnings = [w async for w in tx.execute_and_fetchall('SHOW WARNINGS')]
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
    return plan


def assert_scoped(
    plan: Plan,
    *,
    aliases: Optional[Mapping[str, str]] = None,  # alias -> table, e.g. {'latest_attempt': 'attempts'}
    # per-batch table -> batch column; defaults to the schema's (Plan.batch_scope_columns)
    tables: Optional[Mapping[str, str]] = None,
    allow_filesort: bool = False,
    allow_materialized: bool = False,
):
    """Every access to a per-batch table uses an index on its batch column (never a full table or index scan),
    every subquery is dependent (runs per outer row rather than once over the whole table), nothing is
    materialized, and, unless ``allow_filesort``, nothing is filesorted (a filesort reads every candidate row
    before a ``LIMIT`` can stop it).

    ``EXPLAIN`` reports aliases, not tables, so map each alias the query uses in ``aliases``. A name that is
    neither a real table nor a mapped alias fails: otherwise an aliased scoped table would go unchecked.
    MySQL's own temporary tables (``<derived2>``, ``<subquery3>``, ...) are covered by the materialization check."""
    problems = []
    for a in plan.accesses:
        if a.table.startswith('<'):
            continue
        table = (aliases or {}).get(a.table, a.table)
        name = table if table == a.table else f'{a.table} (alias of {table})'
        if table not in plan.base_tables:
            problems.append(f'{a.table}: not a table; pass its table in aliases at {a.path}')
            continue
        column = (plan.batch_scope_columns if tables is None else tables).get(table)
        if column is None:
            continue
        if a.access_type in ('ALL', 'index'):
            problems.append(f'{name}: full scan (access_type={a.access_type}) at {a.path}')
        elif column not in a.used_key_parts:
            problems.append(f'{name}: key {a.key} used_key_parts {a.used_key_parts} lacks {column} at {a.path}')
    problems.extend(f'non-dependent subquery at {s.path}' for s in plan.subqueries if not s.dependent)
    if not allow_materialized:
        problems.extend(f'materialized subquery at {p}' for p in plan.materialized)
    if not allow_filesort:
        problems.extend(f'filesort at {p}' for p in plan.filesorts)
    assert not problems, '\n'.join(problems) + '\n' + json.dumps(plan.raw, indent=2)


def assert_hint_kept(plan: Plan, hint: str = 'MAX_EXECUTION_TIME'):
    """The optimizer kept ``hint``: it appears in the rewritten statement and no warning was raised
    (an ignored or misplaced hint, e.g. ``MAX_EXECUTION_TIME`` on a subquery, produces a warning)."""
    rewritten = plan.rewritten_sql
    assert rewritten is not None, plan.warnings
    assert f'/*+ {hint}(' in rewritten, rewritten
    bad = [w for w in plan.warnings if w['Level'] != 'Note']
    assert not bad, bad
