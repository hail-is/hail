import time
import datetime
import humanize


def time_msecs():
    return int(time.time() * 1000 + 0.5)


def time_msecs_str(t):
    return datetime.datetime.utcfromtimestamp(t / 1000).strftime(
        '%Y-%m-%dT%H:%M:%SZ')


def humanize_timedelta_msecs(delta_msecs):
    if delta_msecs is None:
        return None

    return humanize.naturaldelta(datetime.timedelta(milliseconds=delta_msecs))
