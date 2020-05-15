def rate_cpu_hour_to_mcpu_msec(rate_cpu_hour):
    rate_cpu_sec = rate_cpu_hour / 3600
    return rate_cpu_sec * 0.001 * 0.001


def rate_gib_hour_to_mib_msec(rate_gib_hour):
    rate_mib_hour = rate_gib_hour / 1024
    rate_mib_sec = rate_mib_hour / 3600
    return rate_mib_sec * 0.001


def rate_gib_month_to_mib_msec(rate_gib_month):
    # average number of days per month = 365.25 / 12 = 30.4375
    avg_n_days_per_month = 30.4375
    rate_gib_hour = rate_gib_month / avg_n_days_per_month / 24
    return rate_gib_hour_to_mib_msec(rate_gib_hour)


def rate_instance_hour_to_fraction_msec(rate_instance_hour, base):
    rate_instance_sec = rate_instance_hour / 3600
    rate_fraction_sec = rate_instance_sec / base
    return rate_fraction_sec * 0.001
