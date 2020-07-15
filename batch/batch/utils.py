import logging
import math

log = logging.getLogger('utils')


def coalesce(x, default):
    if x is not None:
        return x
    return default


def cost_str(cost):
    if cost is None:
        return None
    return f'${cost:.4f}'


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


def adjust_cores_for_memory_request(cores_in_mcpu, memory_in_bytes, worker_type):
    min_cores_mcpu = memory_bytes_to_cores_mcpu(memory_in_bytes, worker_type)
    return max(cores_in_mcpu, min_cores_mcpu)


def adjust_cores_for_packability(cores_in_mcpu):
    cores_in_mcpu = max(1, cores_in_mcpu)
    power = max(-2, math.ceil(math.log2(cores_in_mcpu / 1000)))
    return int(2**power * 1000)
