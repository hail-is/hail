import logging
import json
import secrets
from aiohttp import web
from functools import wraps
from collections import deque

from gear import maybe_parse_bearer_header
from hailtop.utils import secret_alnum_string

from .resource_utils import cost_from_msec_mcpu
from .gcp.instance_config import GCPInstanceConfig

log = logging.getLogger('utils')


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
        self._q = deque()
        self._n_true = 0
        self._seen = set()

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


def instance_config_from_config_dict(config):
    cloud = config.get('cloud', 'gcp')
    assert cloud == 'gcp'
    return GCPInstanceConfig(config)


def instance_config_from_pool_config(pool_config):
    cloud = pool_config.cloud
    assert cloud == 'gcp'
    return GCPInstanceConfig.from_pool_config(pool_config)


async def query_billing_projects(db, user=None, billing_project=None):
    args = []

    where_conditions = ["billing_projects.`status` != 'deleted'"]

    if user:
        where_conditions.append("JSON_CONTAINS(users, JSON_QUOTE(%s))")
        args.append(user)

    if billing_project:
        where_conditions.append('billing_projects.name = %s')
        args.append(billing_project)

    if where_conditions:
        where_condition = f'WHERE {" AND ".join(where_conditions)}'
    else:
        where_condition = ''

    sql = f'''
SELECT billing_projects.name as billing_project,
  billing_projects.`status` as `status`,
  users, msec_mcpu, `limit`, SUM(`usage` * rate) as cost
FROM (
  SELECT billing_project, JSON_ARRAYAGG(`user`) as users
  FROM billing_project_users
  GROUP BY billing_project
  LOCK IN SHARE MODE
) AS t
RIGHT JOIN billing_projects
  ON t.billing_project = billing_projects.name
LEFT JOIN aggregated_billing_project_resources
  ON aggregated_billing_project_resources.billing_project = billing_projects.name
LEFT JOIN resources
  ON resources.resource = aggregated_billing_project_resources.resource
{where_condition}
GROUP BY billing_projects.name, billing_projects.status, msec_mcpu, `limit`
LOCK IN SHARE MODE;
'''

    def record_to_dict(record):
        cost_msec_mcpu = cost_from_msec_mcpu(record['msec_mcpu'])
        cost_resources = record['cost']
        record['accrued_cost'] = coalesce(cost_msec_mcpu, 0) + coalesce(cost_resources, 0)
        del record['msec_mcpu']
        del record['cost']

        if record['users'] is None:
            record['users'] = []
        else:
            record['users'] = json.loads(record['users'])
        return record

    billing_projects = [record_to_dict(record) async for record in db.execute_and_fetchall(sql, tuple(args))]

    return billing_projects
