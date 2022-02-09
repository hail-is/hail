import time
import datetime
import dateutil.parser


def time_msecs() -> int:
    return int(time.time() * 1000 + 0.5)


def time_ns() -> int:
    return time.monotonic_ns()


def time_msecs_str(t) -> str:
    return datetime.datetime.utcfromtimestamp(t / 1000).strftime(
        '%Y-%m-%dT%H:%M:%SZ')


def humanize_timedelta_msecs(delta_msecs):
    import humanize  # pylint: disable=import-outside-toplevel
    if delta_msecs is None:
        return None

    return humanize.naturaldelta(datetime.timedelta(milliseconds=delta_msecs))


def parse_timestamp_msecs(ts):
    if ts is None:
        return ts
    return dateutil.parser.isoparse(ts).timestamp() * 1000
