import time
from typing import Optional, overload, Union
import datetime
import dateutil.parser
from ..humanizex import naturaldelta_msec


def time_msecs() -> int:
    return int(time.time_ns() // 1_000_000)


def time_ns() -> int:
    return time.monotonic_ns()


def time_msecs_str(t: Union[int, float]) -> str:
    return datetime.datetime.utcfromtimestamp(t / 1000).strftime('%Y-%m-%dT%H:%M:%SZ')


@overload
def humanize_timedelta_msecs(delta_msecs: None) -> None: ...


@overload
def humanize_timedelta_msecs(delta_msecs: Union[int, float]) -> str: ...


def humanize_timedelta_msecs(delta_msecs: Optional[Union[int, float]]) -> Optional[str]:
    if delta_msecs is None:
        return None
    return naturaldelta_msec(delta_msecs)


_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
_MS = datetime.timedelta(milliseconds=1)


@overload
def parse_timestamp_msecs(ts: None) -> None: ...


@overload
def parse_timestamp_msecs(ts: str) -> int: ...


def parse_timestamp_msecs(ts: Optional[str]) -> Optional[int]:
    if ts is None:
        return ts
    dt = dateutil.parser.isoparse(ts)
    assert dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None
    return int((dt - _EPOCH) / _MS)
