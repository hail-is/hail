import hail as hl
from hail.expr.expressions.expression_typecheck import *
from hail.expr.functions import _func
from hail.typecheck import *

@typecheck(format=expr_str,
           time=expr_int64,
           zone_id=expr_str)
def strftime(format, time, zone_id):
    return _func("strftime", hl.tstr, format, time, zone_id)

# @typecheck(string=expr_str,
#            format=expr_str,
#            locale=nullable(str))
# def strptime(format, time, locale=None):
#     return _func("strptime", hl.tint64, string, format)
