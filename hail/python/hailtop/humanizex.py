from typing import Union


_MICROSECOND = 1
_MILLISECOND = 1000 * _MICROSECOND
_SECOND = 1000 * _MILLISECOND
_MINUTE = 60 * _SECOND
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR
_WEEK = 7 * _DAY
_MONTH = 30 * _DAY


def _fmt(s: Union[int, float], word: str) -> str:
    s = int(s)
    if s == 1:
        return f'{s} {word}'
    return f'{s} {word}s'


_units = [
    (_MONTH, 'month'),
    (_WEEK, 'week'),
    (_DAY, 'day'),
    (_HOUR, 'hour'),
    (_MINUTE, 'minute'),
    (_SECOND, 'second'),
    (_MILLISECOND, 'millisecond'),
    (_MICROSECOND, 'microsecond'),
]


def naturaldelta(seconds: Union[int, float]) -> str:
    return _naturaldelta(seconds, value_unit = _SECOND, value_name = 'second')


def naturaldelta_msec(milliseconds: Union[int, float]) -> str:
    return _naturaldelta(milliseconds, value_unit = _MILLISECOND, value_name = 'millisecond')


def naturaldelta_usec(microseconds: Union[int, float]) -> str:
    return _naturaldelta(microseconds, value_unit = _MICROSECOND, value_name = 'microsecond')


def _naturaldelta(value: Union[int, float], value_unit: int = _SECOND, value_name: str = 'second') -> str:
    value *= value_unit
    for index in range(len(_units) - 1):
        major_unit, major_name = _units[index]
        minor_unit, minor_name = _units[index + 1]
        if value >= major_unit:
            major = value // major_unit
            minor = (value % major_unit) // minor_unit
            if minor != 0:
                return _fmt(major, major_name) + ' ' + _fmt(minor, minor_name)
            return _fmt(major, major_name)
    return _fmt(value // value_unit, value_name)
