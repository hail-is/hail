import hail as hl
from hail.expr.expressions.expression_typecheck import *
from hail.expr.functions import _func
from hail.typecheck import *

@typecheck(format=expr_str,
           time=expr_int64,
           zone_id=expr_str,
           locale=nullable(expr_str))
def strftime(format, time, zone_id, locale=None):
    return _func("strftime", hl.tstr, format, time, zone_id)

@typecheck(time=expr_str,
           format=expr_str,
           zone_id=expr_str,
           locale=nullable(expr_str))
def strptime(time, format, zone_id, locale=None):
    return _func("strptime", hl.tint64, time, format, zone_id)
