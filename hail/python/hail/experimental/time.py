import hail as hl
from hail.expr.expressions import expr_str, expr_int64
from hail.expr.functions import _func
from hail.typecheck import typecheck


@typecheck(format=expr_str,
           time=expr_int64,
           zone_id=expr_str)
def strftime(format, time, zone_id):
    """
    Convert Unix timestamp to a formatted datetime string.

    Examples
    --------

    >>> hl.eval(hl.experimental.strftime("%Y.%m.%d %H:%M:%S %z", 1562569201, "America/New_York"))
    '2019.07.08 03:00:01 -04:00'

    >>> hl.eval(hl.experimental.strftime("%A, %B %e, %Y. %r", 876541523, "GMT+2"))
    'Saturday, October 11, 1997. 05:45:23 AM'

    >>> hl.eval(hl.experimental.strftime("%A, %B %e, %Y. %r", 876541523, "+08:00"))
    'Saturday, October 11, 1997. 11:45:23 AM'


    Notes
    -----
    The following formatting characters are supported in format strings: A a B b D d e F H I j k l M m n p R r S s T t U u V v W Y y z
    See documentation here: https://linux.die.net/man/3/strftime

    A zone id can take one of three forms. It can be an explicit offset, like "+01:00", a relative offset, like "GMT+2",
    or a IANA timezone database (TZDB) identifier, like "America/New_York". Wikipedia maintains a list of TZDB identifiers here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    Currently, the formatter implicitly uses the "en_US" locale.

    Parameters
    ----------
    format : str or :class:`.Expression` of type :py:data:`.tstr`
        The format string describing how to render the time.
    time : int of :class:`.Expression` of type :py:data:`.tint64`
        A long representing the time as a Unix timestamp.
    zone_id: str or :class:`.Expression` of type :py:data:`.tstr`
        An id representing the timezone. See notes above.

    Returns
    -------
    :class:`~.StringExpression`
        A string of the specified format based on the requested time.
    """
    return _func("strftime", hl.tstr, format, time, zone_id)


@typecheck(time=expr_str,
           format=expr_str,
           zone_id=expr_str)
def strptime(time, format, zone_id):
    """
    Interpret a formatted datetime string as a Unix timestamp (number of seconds since 1970-01-01T00:00Z (ISO)).

    Examples
    --------

    >>> hl.eval(hl.experimental.strptime("07/08/19  3:00:01 AM", "%D %l:%M:%S %p", "America/New_York"))
    1562569201

    >>> hl.eval(hl.experimental.strptime("Saturday, October 11, 1997. 05:45:23 AM", "%A, %B %e, %Y. %r", "GMT+2"))
    876541523

    Notes
    -----
    The following formatting characters are supported in format strings: A a B b D d e F H I j k l M m n p R r S s T t U u V v W Y y z
    See documentation here: https://linux.die.net/man/3/strftime

    A zone id can take one of three forms. It can be an explicit offset, like "+01:00", a relative offset, like "GMT+2",
    or a IANA timezone database (TZDB) identifier, like "America/New_York". Wikipedia maintains a list of TZDB identifiers here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    Currently, the parser implicitly uses the "en_US" locale.

    This function will fail if there is not enough information in the string to determine a particular timestamp.
    For example, if you have the string `"07/08/09"` and the format string `"%Y.%m.%d"`, this method will fail, since that's not specific
    enough to determine seconds from. You can fix this by adding "00:00:00" to your date string and "%H:%M:%S" to your format string.

    Parameters
    ----------
    time : str or :class:`.Expression` of type :py:data:`.tstr`
        The string from which to parse the time.
    format : str or :class:`.Expression` of type :py:data:`.tstr`
        The format string describing how to parse the time.
    zone_id: str or :class:`.Expression` of type :py:data:`.tstr`
        An id representing the timezone. See notes above.

    Returns
    -------
    :class:`~.Int64Expression`
        The Unix timestamp associated with the given time string.
    """
    return _func("strptime", hl.tint64, time, format, zone_id)
