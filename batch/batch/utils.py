import logging
import math
import json
from collections import deque

from hailtop.utils import is_power_of_two
from .globals import RESERVED_STORAGE_GB_PER_CORE

log = logging.getLogger('utils')


def round_up_division(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def coalesce(x, default):
    if x is not None:
        return x
    return default


def cost_from_msec_mcpu(msec_mcpu):
    if msec_mcpu is None:
        return None

    worker_type = 'standard'
    worker_cores = 16
    worker_disk_size_gb = 100

    # https://cloud.google.com/compute/all-pricing

    # per instance costs
    # persistent SSD: $0.17 GB/month
    # average number of days per month = 365.25 / 12 = 30.4375
    avg_n_days_per_month = 30.4375

    disk_cost_per_instance_hour = 0.17 * worker_disk_size_gb / avg_n_days_per_month / 24

    ip_cost_per_instance_hour = 0.004

    instance_cost_per_instance_hour = disk_cost_per_instance_hour + ip_cost_per_instance_hour

    # per core costs
    if worker_type == 'standard':
        cpu_cost_per_core_hour = 0.01
    elif worker_type == 'highcpu':
        cpu_cost_per_core_hour = 0.0075
    else:
        assert worker_type == 'highmem'
        cpu_cost_per_core_hour = 0.0125

    service_cost_per_core_hour = 0.01

    total_cost_per_core_hour = (
        cpu_cost_per_core_hour
        + instance_cost_per_instance_hour / worker_cores
        + service_cost_per_core_hour)

    return (msec_mcpu * 0.001 * 0.001) * (total_cost_per_core_hour / 3600)


def worker_memory_per_core_gb(worker_type):
    if worker_type == 'standard':
        m = 3.75
    elif worker_type == 'highmem':
        m = 6.5
    else:
        assert worker_type == 'highcpu', worker_type
        m = 0.9
    return m


def worker_memory_per_core_bytes(worker_type):
    m = worker_memory_per_core_gb(worker_type)
    return int(m * 1024**3)


def memory_bytes_to_cores_mcpu(memory_in_bytes, worker_type):
    return math.ceil((memory_in_bytes / worker_memory_per_core_bytes(worker_type)) * 1000)


def cores_mcpu_to_memory_bytes(cores_in_mcpu, worker_type):
    return int((cores_in_mcpu / 1000) * worker_memory_per_core_bytes(worker_type))


def unreserved_worker_data_disk_size_gib(worker_local_ssd_data_disk, worker_pd_ssd_data_disk_size_gib, worker_cores):
    reserved_image_size = 20
    reserved_container_size = RESERVED_STORAGE_GB_PER_CORE * worker_cores
    if worker_local_ssd_data_disk:
        # local ssd is 375Gi
        # reserve 20Gi for images
        return 375 - reserved_image_size - reserved_container_size
    return worker_pd_ssd_data_disk_size_gib - reserved_image_size - reserved_container_size


def adjust_cores_for_packability(cores_in_mcpu):
    cores_in_mcpu = max(1, cores_in_mcpu)
    power = max(-2, math.ceil(math.log2(cores_in_mcpu / 1000)))
    return int(2**power * 1000)


def is_valid_core_count(cores_mcpu, worker_cores):
    cores = cores_mcpu / 1000
    if not is_power_of_two(cores):
        return False
    power = math.ceil(math.log2(cores))
    return -2 <= power <= math.log2(worker_cores)


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
GROUP BY billing_projects.name, billing_projects.status, users, msec_mcpu, `limit`
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

    billing_projects = [record_to_dict(record)
                        async for record
                        in db.execute_and_fetchall(sql, tuple(args))]

    return billing_projects
