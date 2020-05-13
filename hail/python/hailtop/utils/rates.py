def rate_cpu_hour_to_mcpu_msec(rate_cpu_hour):
    rate_cpu_sec = rate_cpu_hour / 3600
    return rate_cpu_sec * 0.001 * 0.001


def rate_gb_hour_to_mb_msec(rate_gb_hour):
    rate_mb_hour = rate_gb_hour / 1024
    rate_mb_sec = rate_mb_hour / 3600
    return rate_mb_sec * 0.001


def rate_gb_month_to_mb_msec(rate_gb_month):
    # average number of days per month = 365.25 / 12 = 30.4375
    avg_n_days_per_month = 30.4375
    rate_gb_hour = rate_gb_month / avg_n_days_per_month / 24
    return rate_gb_hour_to_mb_msec(rate_gb_hour)


def rate_instance_hour_to_fraction_msec(rate_instance_hour, base):
    rate_instance_sec = rate_instance_hour / 3600
    rate_fraction_sec = rate_instance_sec / base
    return rate_fraction_sec * 0.001
