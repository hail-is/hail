import datetime
import re
from typing import Dict, Optional

rfc3339_re = re.compile(
    # https://www.rfc-editor.org/rfc/rfc3339#section-5.6
    '([0-9][0-9][0-9][0-9])'  # YYYY
    '-'
    '([0-9][0-9])'  # MM
    '-'
    '([0-9][0-9])'  # DD
    '[Tt ]'  # see NOTE in link
    '([0-9][0-9])'  # HH
    ':'
    '([0-9][0-9])'  # MM
    ':'
    '([0-9][0-9])'  # SS
    '(.[0-9][0-9]*)?'  # optional fractional seconds
    '([Zz]|[+-][0-9][0-9]:[0-9][0-9])'  # offset / timezone
)
_timezone_cache: Dict[str, datetime.timezone] = {}


def parse_rfc3339(s: str) -> datetime.datetime:
    parts = rfc3339_re.fullmatch(s)
    if parts is None:
        raise ValueError(f'Datetime string is not RFC3339 compliant: {s}.')
    year, month, day, hour, minute, second, fractional_second, offset = parts.groups()

    tz: Optional[datetime.timezone]
    if offset in ('z', 'Z'):
        tz = datetime.timezone.utc
    else:
        tz = _timezone_cache.get(offset)
        if tz is None:
            if offset[0] not in ('+', '-') or offset[3] != ':':
                raise ValueError(f'Datetime string is not RFC3339 compliant: {s}.')
            is_neg = offset[0] == '-'
            offset_h = int(offset[1:3])
            offset_m = int(offset[4:6])
            if is_neg:
                offset_h = offset_h * -1
                offset_m = offset_m * -1
            tz_delta = datetime.timedelta(hours=offset_h, minutes=offset_m)
            tz = datetime.timezone(tz_delta)
            _timezone_cache[offset] = tz

    if fractional_second is None:
        microsecond = 0
    else:
        fractional_second = fractional_second[1:]  # skip the '.'
        n_digits = len(fractional_second)
        if n_digits > 6:
            # python only supports integral microseconds
            rounding_digit = fractional_second[6]
            microsecond = int(fractional_second[:6])
            if rounding_digit in ('5', '6', '7', '8', '9'):
                microsecond += 1
        else:
            fractional_second = int(fractional_second)
            magnitude_relative_to_micro = 10 ** (6 - n_digits)
            microsecond = int(fractional_second) * magnitude_relative_to_micro

    return datetime.datetime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
        microsecond=microsecond,
        tzinfo=tz,
    )
