import time
from typing import Optional, overload
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


@overload
def parse_timestamp_msecs(ts: None) -> None: ...
@overload
def parse_timestamp_msecs(ts: str) -> int: ...
def parse_timestamp_msecs(ts: Optional[str]) -> Optional[int]:
    if ts is None:
        return ts
    return int(dateutil.parser.isoparse(ts).timestamp() * 1000)
