import json
import logging
import secrets
from collections import deque
from functools import wraps
from typing import Deque, Optional, Set, Tuple

from aiohttp import web

from gear import Database, maybe_parse_bearer_header
from hailtop.utils import secret_alnum_string

log = logging.getLogger('utils')


@web.middleware
async def unavailable_if_frozen(request: web.Request, handler):
    if request.method in ("POST", "PATCH") and request.app['frozen']:
        raise web.HTTPServiceUnavailable()
    return await handler(request)


def authorization_token(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    session_id = maybe_parse_bearer_header(auth_header)
    if not session_id:
        return None
    return session_id


def batch_only(fun):
    @wraps(fun)
    async def wrapped(request):
        token = authorization_token(request)
        if not token:
            raise web.HTTPUnauthorized()

        if not secrets.compare_digest(token, request.app['internal_token']):
            raise web.HTTPUnauthorized()

        return await fun(request)

    return wrapped


def coalesce(x, default):
    if x is not None:
        return x
    return default


class Box:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'{self.value}'


class WindowFractionCounter:
    def __init__(self, window_size: int):
        self._window_size = window_size
        self._q: Deque[Tuple[str, bool]] = deque()
        self._n_true = 0
        self._seen: Set[str] = set()

    def clear(self):
        self._q.clear()
        self._n_true = 0
        self._seen = set()

    def push(self, key: str, x: bool):
        self.assert_valid()
        if key in self._seen:
            return
        while len(self._q) >= self._window_size:
            old_key, old = self._q.popleft()
            self._seen.remove(old_key)
            if old:
                self._n_true -= 1
        self._q.append((key, x))
        self._seen.add(key)
        if x:
            self._n_true += 1
        self.assert_valid()

    def fraction(self) -> float:
        self.assert_valid()
        # (1, 1) prior
        return (self._n_true + 1) / (len(self._q) + 2)

    def __repr__(self):
        self.assert_valid()
        return f'{self._n_true}/{len(self._q)}'

    def assert_valid(self):
        assert len(self._q) <= self._window_size
        assert 0 <= self._n_true <= self._window_size


class ExceededSharesCounter:
    def __init__(self):
        self._global_counter = WindowFractionCounter(10)

    def push(self, success: bool):
        token = secret_alnum_string(6)
        self._global_counter.push(token, success)

    def rate(self) -> float:
        return self._global_counter.fraction()

    def __repr__(self):
        return f'global {self._global_counter}'


async def query_billing_projects(db, user=None, billing_project=None):
    args = []

    where_conditions = ["billing_projects.`status` != 'deleted'"]

    if user:
        where_conditions.append("JSON_CONTAINS(users, JSON_QUOTE(%s))")
        args.append(user)

    if billing_project:
        where_conditions.append('billing_projects.name_cs = %s')
        args.append(billing_project)

    if where_conditions:
        where_condition = f'WHERE {" AND ".join(where_conditions)}'
    else:
        where_condition = ''

    sql = f'''
WITH base_t AS (
SELECT billing_projects.name as billing_project,
  billing_projects.`status` as `status`,
  users, `limit`
FROM (
  SELECT billing_project, JSON_ARRAYAGG(`user_cs`) as users
  FROM billing_project_users
  GROUP BY billing_project
  LOCK IN SHARE MODE
) AS t
RIGHT JOIN billing_projects
  ON t.billing_project = billing_projects.name
{where_condition}
GROUP BY billing_projects.name, billing_projects.status, `limit`
LOCK IN SHARE MODE
)
SELECT base_t.*, COALESCE(SUM(`usage` * rate), 0) as accrued_cost
FROM base_t
LEFT JOIN (
  SELECT base_t.billing_project, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM base_t
  LEFT JOIN aggregated_billing_project_user_resources_v2
    ON base_t.billing_project = aggregated_billing_project_user_resources_v2.billing_project
  GROUP BY base_t.billing_project, resource_id
) AS usage_t ON usage_t.billing_project = base_t.billing_project
LEFT JOIN resources ON resources.resource_id = usage_t.resource_id
GROUP BY base_t.billing_project;
'''

    def record_to_dict(record):
        if record['users'] is None:
            record['users'] = []
        else:
            record['users'] = json.loads(record['users'])
        return record

    billing_projects = [record_to_dict(record) async for record in db.execute_and_fetchall(sql, tuple(args))]

    return billing_projects


def json_to_value(x):
    if x is None:
        return x
    return json.loads(x)


def regions_to_bits_rep(selected_regions, all_regions_mapping):
    result = 0
    for region in selected_regions:
        idx = all_regions_mapping[region]
        assert idx < 64, str(all_regions_mapping)
        result |= 1 << (idx - 1)
    return result


def regions_bits_rep_to_regions(regions_bits_rep, all_regions_mapping):
    if regions_bits_rep is None:
        return None
    result = []
    for region, idx in all_regions_mapping.items():
        selected_region = bool((regions_bits_rep >> idx - 1) & 1)
        if selected_region:
            result.append(region)
    return result


async def compact_agg_billing_project_users_table(db: Database):
    async def compact(key: Optional[Tuple[str, str, int]]) -> Optional[Tuple[str, str, int]]:
        if key:
            billing_project, user, resource_id = key
            where_condition = (
                'AND ((billing_project > %s) OR'
                '(billing_project = %s AND `user` > %s) OR'
                '(billing_project = %s AND `user` = %s AND resource_id > %s))'
            )
            args = [billing_project, billing_project, user, billing_project, user, resource_id]
        else:
            where_condition = ''
            args = []

        token = secret_alnum_string(5)
        scratch_table_name = f'`scratch_{token}`'

        try:
            next_key = await db.execute_and_fetchone(
                f'''
CREATE TEMPORARY TABLE {scratch_table_name} ENGINE=MEMORY
AS (SELECT billing_project, `user`, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
    FROM aggregated_billing_project_user_resources_v2
    WHERE token != 0 {where_condition}
    GROUP BY billing_project, `user`, resource_id
    ORDER BY billing_project, `user`, resource_id
    LIMIT 2
    LOCK IN SHARE MODE
);

DELETE aggregated_billing_project_user_resources_v2 FROM aggregated_billing_project_user_resources_v2
INNER JOIN {scratch_table_name} ON aggregated_billing_project_user_resources_v2.billing_project = {scratch_table_name}.billing_project AND
                      aggregated_billing_project_user_resources_v2.`user` = {scratch_table_name}.`user` AND
                      aggregated_billing_project_user_resources_v2.resource_id = {scratch_table_name}.resource_id
WHERE token != 0;

INSERT INTO aggregated_billing_project_user_resources_v2 (billing_project, `user`, resource_id, token, `usage`)
SELECT billing_project, `user`, resource_id, 0, `usage`
FROM {scratch_table_name}
ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_v2.`usage` + {scratch_table_name}.`usage`;

SELECT billing_project, `user`, resource_id
FROM {scratch_table_name}
ORDER BY billing_project DESC, `user` DESC, resource_id DESC
LIMIT 1;
''',
                args,
            )
        finally:
            await db.just_execute(f'DROP TEMPORARY TABLE IF EXISTS {scratch_table_name};')

        if next_key is None:
            return None
        return (next_key['billing_project'], next_key['user'], next_key['resource_id'])

    next_key = await compact(None)
    while next_key is not None:
        await compact(next_key)


async def compact_agg_billing_project_users_by_date_table(db: Database):
    async def compact(key: Optional[Tuple[str, str, str, int]]) -> Optional[Tuple[str, str, str, int]]:
        if key:
            billing_date, billing_project, user, resource_id = key
            where_condition = (
                'AND ((billing_date > %s) OR'
                '(billing_date = %s AND billing_project > %s) OR'
                '(billing_date = %s AND billing_project = %s AND `user` > %s) OR'
                '(billing_date = %s AND billing_project = %s AND `user` = %s AND resource_id > %s))'
            )
            args = [
                billing_date,
                billing_date,
                billing_project,
                billing_date,
                billing_project,
                user,
                billing_date,
                billing_project,
                user,
                resource_id,
            ]
        else:
            where_condition = ''
            args = []

        token = secret_alnum_string(5)
        scratch_table_name = f'`scratch_{token}`'

        try:
            next_key = await db.execute_and_fetchone(
                f'''
CREATE TEMPORARY TABLE {scratch_table_name} ENGINE=MEMORY
AS (SELECT billing_date, billing_project, `user`, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
    FROM aggregated_billing_project_user_resources_by_date_v2
    WHERE token != 0 {where_condition}
    GROUP BY billing_date, billing_project, `user`, resource_id
    ORDER BY billing_date, billing_project, `user`, resource_id
    LIMIT 2
    LOCK IN SHARE MODE
);

DELETE aggregated_billing_project_user_resources_by_date_v2 FROM aggregated_billing_project_user_resources_by_date_v2
INNER JOIN {scratch_table_name} ON aggregated_billing_project_user_resources_by_date_v2.billing_date = {scratch_table_name}.billing_date AND
                      aggregated_billing_project_user_resources_by_date_v2.billing_project = {scratch_table_name}.billing_project AND
                      aggregated_billing_project_user_resources_by_date_v2.`user` = {scratch_table_name}.`user` AND
                      aggregated_billing_project_user_resources_by_date_v2.resource_id = {scratch_table_name}.resource_id
WHERE token != 0;

INSERT INTO aggregated_billing_project_user_resources_by_date_v2 (billing_date, billing_project, `user`, resource_id, token, `usage`)
SELECT billing_date, billing_project, `user`, resource_id, 0, `usage`
FROM {scratch_table_name}
ON DUPLICATE KEY UPDATE `usage` = aggregated_billing_project_user_resources_by_date_v2.`usage` + {scratch_table_name}.`usage`;

SELECT billing_date, billing_project, `user`, resource_id
FROM {scratch_table_name}
ORDER BY billing_date DESC, billing_project DESC, `user` DESC, resource_id DESC
LIMIT 1;
''',
                args,
            )
        finally:
            await db.just_execute(f'DROP TEMPORARY TABLE IF EXISTS {scratch_table_name};')

        if next_key is None:
            return None
        return (next_key['billing_date'], next_key['billing_project'], next_key['user'], next_key['resource_id'])

    next_key = await compact(None)
    while next_key is not None:
        await compact(next_key)
