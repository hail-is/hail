import re

COMPUTE_RESOURCE_REGEX = re.compile('compute/([^-/]+)-([^-/]+)-([^-/]+)/(\\d+)')
MEMORY_RESOURCE_REGEX = re.compile('memory/([^-]+)-([^-]+)-([^-]+)/(\\d+)')
BOOT_DISK_RESOURCE_REGEX = re.compile('boot-disk/([^/]+)/(\\d+)')
IP_FEE_RESOURCE_REGEX = re.compile('ip-fee/([^/]+)/(\\d+)')
SERVICE_FEE_RESOURCE_REGEX = re.compile('service-fee/(\\d+)')

# https://cloud.google.com/compute/all-pricing
RATES = {
    'compute/n1-standard-preemptible/1': 0.006655,  # cpu-hour
    'memory/n1-standard-preemptible/1': 0.000892,  # gb-hour
    'boot-disk/pd-ssd/1': 0.17,  # gi-month
    'ip-fee/1024/1': 0.004,  # instance-hour
    'service-fee/1': 0.01  # cpu-hour
}


def cost_from_resources(resources):
    if resources is None:
        return None

    total_cost = 0
    for resource_name, usage in resources.items():
        rate = RATES[resource_name]

        if COMPUTE_RESOURCE_REGEX.fullmatch(resource_name):
            # rate is in units of cpu-hour
            # usage is in units of msec-mcpu
            cost = rate / 3600 * (usage * 0.001 * 0.001)
        elif MEMORY_RESOURCE_REGEX.fullmatch(resource_name):
            # rate is in units of gb-hour
            # usage is in units of msec-mb
            cost = rate / 3600 * (usage * 0.001 * 0.001)
        elif BOOT_DISK_RESOURCE_REGEX.fullmatch(resource_name):
            # rate is in units of gi-month
            # usage is in units of msec-mb
            # average number of days per month = 365.25 / 12 = 30.4375
            avg_n_days_per_month = 30.4375
            cost = rate / avg_n_days_per_month / 24 / 3600 * (usage * 0.001 * (1 / 1024))
        elif IP_FEE_RESOURCE_REGEX.fullmatch(resource_name):
            # rate is in units of instance-hour
            # usage is an integer representing a fraction of an instance - msec
            denominator, _ = IP_FEE_RESOURCE_REGEX.fullmatch(resource_name).groups()
            denominator = int(denominator)
            cost = rate / 3600 * (usage * 0.001 / denominator)
        elif SERVICE_FEE_RESOURCE_REGEX.fullmatch(resource_name):
            # rate is in units of cpu-hour
            # usage is in units of msec-mcpu
            cost = rate / 3600 * (usage * 0.001 * 0.001)
        else:
            raise ValueError(resource_name)
        total_cost += cost
    return total_cost
