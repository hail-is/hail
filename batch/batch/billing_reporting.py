"""Read-only billing spend queries with access-control scoping.

This module handles querying accrued cost and billing history. It does not mutate any billing
entities — see billing_project_management.py for that.
"""

import json
from typing import Any, Dict, List, Optional

from gear import Database


async def query_billing_projects_with_cost(db, user=None, billing_project=None, status=None) -> List[Dict[str, Any]]:
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
  users, `limit`, COALESCE(SUM(agg.`usage` * resources.rate), 0) AS accrued_cost
FROM billing_projects
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
GROUP BY billing_projects.name, billing_projects.`status`, billing_projects.`limit`, users;
"""

    billing_projects = []
    async for record in db.select_and_fetchall(sql, tuple(args)):
        record['users'] = json.loads(record['users']) if record['users'] is not None else []
        billing_projects.append(record)

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
  users, `limit`
FROM billing_projects
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

    return billing_projects
