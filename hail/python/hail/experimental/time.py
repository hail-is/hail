import hail as hl
from hail.expr.expressions.expression_typecheck import *
from hail.expr.functions import _func
from hail.typecheck import *

@typecheck(format=expr_str,
           time=expr_int64,
           zone_id=expr_str)
def strftime(format, time, zone_id):
    """
    Convert Unix timestamp to a formatted datetime string.

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
    """
    return _func("strftime", hl.tstr, format, time, zone_id)

@typecheck(time=expr_str,
           format=expr_str,
           zone_id=expr_str)
def strptime(time, format, zone_id):
    """
    Interpret a formatted datetime string as a Unix timestamp.

    Notes
    -----
    The following formatting characters are supported in format strings: A a B b D d e F H I j k l M m n p R r S s T t U u V v W Y y z
    See documentation here: https://linux.die.net/man/3/strftime

    A zone id can take one of three forms. It can be an explicit offset, like "+01:00", a relative offset, like "GMT+2",
    or a IANA timezone database (TZDB) identifier, like "America/New_York". Wikipedia maintains a list of TZDB identifiers here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

    Currently, the parser implicitly uses the "en_US" locale.

    Parameters
    ----------
    format : str or :class:`.Expression` of type :py:data:`.tstr`
        The format string describing how to parse the time.
    time : int of :class:`.Expression` of type :py:data:`.tint64`
        A long representing the time as a Unix timestamp.
    zone_id: str or :class:`.Expression` of type :py:data:`.tstr`
        An id representing the timezone. See notes above.
    """
    return _func("strptime", hl.tint64, time, format, zone_id)
