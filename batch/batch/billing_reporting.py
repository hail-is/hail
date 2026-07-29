"""Read-only billing spend queries with access-control scoping.

This module handles querying accrued cost and billing history. It does not
mutate any billing entities — see billing_project_management.py for that.

Access control tiers:
  global_bm        — no filter, sees all spend across all billing projects
  quote_manager    — sees all spend in billing projects under their quotes,
                     plus their own spend elsewhere
  regular user     — sees only their own spend
"""

import datetime
import json
from typing import Any, Dict, List, Optional

from gear import Database


async def query_billing_projects_with_cost(
    db,
    user=None,
    billing_project=None,
    status=None,
    quote_manager_user=None,
) -> List[Dict[str, Any]]:
    where_conditions = ["billing_projects.`status` != 'deleted'"]
    args = []

    if user and quote_manager_user:
        where_conditions.append(
            "(JSON_CONTAINS(users, JSON_QUOTE(%s)) OR EXISTS ("
            "SELECT 1 FROM quote_managers qm "
            "WHERE qm.quote_id = billing_projects.quote_id AND qm.user = %s))"
        )
        args.append(user)
        args.append(quote_manager_user)
    elif user:
        where_conditions.append("JSON_CONTAINS(users, JSON_QUOTE(%s))")
        args.append(user)
    elif quote_manager_user:
        where_conditions.append(
            "EXISTS (SELECT 1 FROM quote_managers qm WHERE qm.quote_id = billing_projects.quote_id AND qm.user = %s)"
        )
        args.append(quote_manager_user)

    if billing_project:
        where_conditions.append('billing_projects.name_cs = %s')
        args.append(billing_project)

    if status:
        where_conditions.append('billing_projects.`status` = %s')
        args.append(status)

    if where_conditions:
        where_condition = f'WHERE {" AND ".join(where_conditions)}'
    else:
        where_condition = ''

    sql = f"""
SELECT billing_projects.name as billing_project,
  billing_projects.`status` as `status`,
  users,
  billing_projects.`limit`,
  billing_projects.quote_id,
  q.name AS quote_name,
  billing_projects.low_budget_alert,
  billing_projects.description,
  IF(billing_projects.`limit` IS NULL, NULL, billing_projects.`limit` - COALESCE(SUM(agg.`usage` * resources.rate), 0)) AS remaining,
  COALESCE(SUM(agg.`usage` * resources.rate), 0) AS accrued_cost
FROM billing_projects
LEFT JOIN quotes q ON q.id = billing_projects.quote_id
LEFT JOIN LATERAL (
  SELECT billing_project, JSON_ARRAYAGG(`user_cs`) as users
  FROM billing_project_users
  WHERE billing_project_users.billing_project = billing_projects.name
  GROUP BY billing_project_users.billing_project
) AS t ON TRUE
LEFT JOIN aggregated_billing_project_user_resources_v3 as agg
  ON billing_projects.name = agg.billing_project
LEFT JOIN resources ON resources.resource_id = agg.resource_id
{where_condition}
GROUP BY billing_projects.name, billing_projects.`status`, billing_projects.`limit`,
  billing_projects.quote_id, q.name, billing_projects.low_budget_alert, billing_projects.description, users;
"""

    billing_projects = []
    async for record in db.select_and_fetchall(sql, tuple(args)):
        record['users'] = json.loads(record['users']) if record['users'] is not None else []
        billing_projects.append(record)

    quote_ids = list({bp['quote_id'] for bp in billing_projects if bp['quote_id'] is not None})
    qms_by_quote_id: Dict[int, List[Dict]] = {}
    if quote_ids:
        placeholders = ', '.join(['%s'] * len(quote_ids))
        async for row in db.select_and_fetchall(
            f'SELECT qm.quote_id, qm.user, qm.role, q.name AS quote_name'
            f' FROM quote_managers qm JOIN quotes q ON q.id = qm.quote_id'
            f' WHERE qm.quote_id IN ({placeholders})',
            tuple(quote_ids),
        ):
            qms_by_quote_id.setdefault(row['quote_id'], []).append(row)

    for bp in billing_projects:
        merged: Dict[str, List[str]] = {}
        for username in bp['users']:
            merged.setdefault(username, []).append(f'{bp["billing_project"]}:member')
        qms_for_bp = qms_by_quote_id.get(bp['quote_id'], [])
        for qm in qms_for_bp:
            merged.setdefault(qm['user'], []).append(f'{qm["quote_name"]}:{qm["role"]}')
        bp['users'] = [{'user': u, 'roles': roles} for u, roles in merged.items()]
        if quote_manager_user is None:
            bp['can_view_quote'] = True
        else:
            bp['can_view_quote'] = any(qm['user'] == quote_manager_user for qm in qms_for_bp)

    return billing_projects


async def query_billing_projects_without_cost(
    db: Database, user: Optional[str] = None, billing_project: Optional[str] = None, status: Optional[str] = None
) -> List[Dict[str, Any]]:
    where_conditions = ["billing_projects.`status` != 'deleted'"]
    args = []

    if user:
        where_conditions.append("JSON_CONTAINS(users, JSON_QUOTE(%s))")
        args.append(user)

    if billing_project:
        where_conditions.append('billing_projects.name_cs = %s')
        args.append(billing_project)

    if status:
        where_conditions.append('billing_projects.`status` = %s')
        args.append(status)

    if where_conditions:
        where_condition = f'WHERE {" AND ".join(where_conditions)}'
    else:
        where_condition = ''

    sql = f"""
SELECT billing_projects.name as billing_project,
  billing_projects.`status` as `status`,
  billing_projects.quote_id,
  q.name AS quote_name,
  users, `limit`
FROM billing_projects
LEFT JOIN quotes q ON q.id = billing_projects.quote_id
LEFT JOIN LATERAL (
  SELECT billing_project, JSON_ARRAYAGG(`user_cs`) as users
  FROM billing_project_users
  WHERE billing_project_users.billing_project = billing_projects.name
  GROUP BY billing_project_users.billing_project
) AS t ON TRUE
{where_condition};
"""

    billing_projects = []
    async for record in db.select_and_fetchall(sql, tuple(args)):
        record['users'] = json.loads(record['users']) if record['users'] is not None else []
        billing_projects.append(record)

    quote_ids = list({bp['quote_id'] for bp in billing_projects if bp['quote_id'] is not None})
    qms_by_quote_id: Dict[int, List[Dict]] = {}
    if quote_ids:
        placeholders = ', '.join(['%s'] * len(quote_ids))
        async for row in db.select_and_fetchall(
            f'SELECT qm.quote_id, qm.user, qm.role, q.name AS quote_name'
            f' FROM quote_managers qm JOIN quotes q ON q.id = qm.quote_id'
            f' WHERE qm.quote_id IN ({placeholders})',
            tuple(quote_ids),
        ):
            qms_by_quote_id.setdefault(row['quote_id'], []).append(row)

    for bp in billing_projects:
        merged: Dict[str, List[str]] = {}
        for username in bp['users']:
            merged.setdefault(username, []).append(f'{bp["billing_project"]}:member')
        for qm in qms_by_quote_id.get(bp['quote_id'], []):
            merged.setdefault(qm['user'], []).append(f'{qm["quote_name"]}:{qm["role"]}')
        bp['users'] = [{'user': u, 'roles': roles} for u, roles in merged.items()]

    return billing_projects


async def query_billing_history(
    db: Database,
    start: datetime.datetime,
    end: Optional[datetime.datetime],
    user: Optional[str] = None,
    quote_manager_user: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return per-(billing_project, user) spend for the given date range.

    Access control:
      user=None, quote_manager_user=None  — global_bm, no filter
      user=X, quote_manager_user=X        — quote manager: own spend + all
                                            spend in managed quotes
      user=X, quote_manager_user=None     — regular user: own spend only
    """
    where_conditions = [
        "billing_projects.`status` != 'deleted'",
        "billing_date >= %s",
    ]
    args: List[Any] = [start]

    if end is not None:
        where_conditions.append("billing_date <= %s")
        args.append(end)

    if user is not None:
        if quote_manager_user is not None:
            where_conditions.append(
                "(`user` = %s OR billing_projects.quote_id IN (SELECT quote_id FROM quote_managers WHERE `user` = %s))"
            )
            args.extend([user, quote_manager_user])
        else:
            where_conditions.append("`user` = %s")
            args.append(user)

    sql = f"""
SELECT
  billing_project,
  `user`,
  quote_name,
  COALESCE(SUM(`usage` * rate), 0) AS cost
FROM (
  SELECT billing_project, `user`, resource_id, quotes.name AS quote_name,
    CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_billing_project_user_resources_by_date_v3
  LEFT JOIN billing_projects ON billing_projects.name = aggregated_billing_project_user_resources_by_date_v3.billing_project
  LEFT JOIN quotes ON quotes.id = billing_projects.quote_id
  WHERE {' AND '.join(where_conditions)}
  GROUP BY billing_project, `user`, resource_id, quote_name
) AS t
LEFT JOIN resources ON resources.resource_id = t.resource_id
GROUP BY billing_project, `user`, quote_name;
"""

    return [record async for record in db.select_and_fetchall(sql, args)]


async def query_billing_breakdown(
    db: Database,
    start: datetime.datetime,
    end: datetime.datetime,
    user: Optional[str] = None,
    quote_manager_user: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return per-(billing_project, user, resource) spend for the given date range.

    Same access-control semantics as query_billing_history.
    """
    where_conditions = [
        "billing_projects.`status` != 'deleted'",
        'billing_date >= %s',
        'billing_date <= %s',
    ]
    args: List[Any] = [start, end]

    if user is not None:
        if quote_manager_user is not None:
            where_conditions.append(
                "(`user` = %s OR billing_projects.quote_id IN (SELECT quote_id FROM quote_managers WHERE `user` = %s))"
            )
            args.extend([user, quote_manager_user])
        else:
            where_conditions.append("`user` = %s")
            args.append(user)

    sql = f"""
SELECT billing_project, `user`, resources.resource, COALESCE(SUM(`usage` * rate), 0) AS cost
FROM (
  SELECT billing_project, `user`, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_billing_project_user_resources_by_date_v3
  LEFT JOIN billing_projects ON billing_projects.name = aggregated_billing_project_user_resources_by_date_v3.billing_project
  WHERE {' AND '.join(where_conditions)}
  GROUP BY billing_project, `user`, resource_id
) AS t
LEFT JOIN resources ON resources.resource_id = t.resource_id
GROUP BY billing_project, `user`, resources.resource
HAVING cost > 0;
"""

    return [
        {
            'billing_project': r['billing_project'],
            'user': r['user'],
            'resource': r['resource'],
            'cost': float(r['cost']),
        }
        async for r in db.select_and_fetchall(sql, args)
    ]
