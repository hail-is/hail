import json
import logging
import secrets
from collections import deque
from functools import wraps
from typing import Deque, Set, Tuple

from aiohttp import web

from gear import maybe_parse_bearer_header
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
SELECT billing_projects.name as billing_project,
  billing_projects.`status` as `status`,
  users, `limit`, COALESCE(cost_t.cost, 0) AS accrued_cost
FROM billing_projects
LEFT JOIN LATERAL (
  SELECT billing_project, JSON_ARRAYAGG(`user_cs`) as users
  FROM billing_project_users
  WHERE billing_project_users.billing_project = billing_projects.name
  GROUP BY billing_project_users.billing_project
  LOCK IN SHARE MODE
) AS t ON TRUE
LEFT JOIN LATERAL (
  SELECT SUM(`usage` * rate) as cost
  FROM (
    SELECT billing_project, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
    FROM aggregated_billing_project_user_resources_v2
    WHERE billing_projects.name = aggregated_billing_project_user_resources_v2.billing_project
    GROUP BY billing_project, resource_id
  ) AS usage_t
  LEFT JOIN resources ON resources.resource_id = usage_t.resource_id
  GROUP BY usage_t.billing_project
) AS cost_t ON TRUE
{where_condition}
LOCK IN SHARE MODE;
'''

    def record_to_dict(record):
        if record['users'] is None:
            record['users'] = []
        else:
            record['users'] = json.loads(record['users'])
        return record

    log.info(f'sql {sql} args {args}')

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
