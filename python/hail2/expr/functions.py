from __future__ import print_function  # Python 2 and 3 print compatibility

from hail2.expr.column import *
from hail.typ import *
from hail.typecheck import *

integral = oneof(Int32Column, Int64Column, integral)
numeric = oneof(Float32Column, Float64Column, float, integral)
list_typ = oneof(list, ArrayColumn)
bool_typ = oneof(bool, BooleanColumn)
struct_typ = oneof(Struct, StructColumn)
string_typ = oneof(str, StringColumn)
variant_typ = oneof(Variant, VariantColumn)


def _func(name, ret_typ, *args):
    @args_to_expr
    def _(*args):
        return convert_column(Column("{name}({args})".format(name=name, args=", ".join(args)), ret_typ))
    return _(*args)


@typecheck(c1=integral, c2=integral, c3=integral, c4=integral)
def chisq(c1, c2, c3, c4):
    ret_typ = TStruct(['pValue', 'oddsRatio'], [TFloat64(), TFloat64()])
    return _func("chisq", ret_typ, c1, c2, c3, c4)


@typecheck(left=variant_typ, right=variant_typ)
def combine_variants(left, right):
    ret_typ = TStruct(['variant', 'laIndices', 'raIndices'], [TVariant(), TDict(TInt32(), TInt32()), TDict(TInt32(), TInt32())])
    return _func("combineVariants", ret_typ, left, right)


@typecheck(c1=integral, c2=integral, c3=integral, c4=integral, min_cell_count=integral)
def ctt(c1, c2, c3, c4, min_cell_count):
    ret_typ = TStruct(['pValue', 'oddsRatio'], [TFloat64(), TFloat64()])
    return _func("ctt", ret_typ, c1, c2, c3, c4, min_cell_count)


@typecheck(keys=list_typ, values=list_typ)
def Dict(keys, values):
    key_typ = get_typ(keys)
    value_typ = get_typ(values)
    ret_typ = TDict(key_typ, value_typ)
    return _func("Dict", ret_typ, keys, values)


@typecheck(x=numeric, lamb=numeric, logP=bool_typ)
def dpois(x, lamb, logP=False):
    return _func("dpois", TFloat64(), x, lamb, logP)


@typecheck(s=StructColumn, identifiers=tupleof(string_typ))
def drop(s, *identifiers):  # FIXME: Need to be able to take regular Struct as input args
    expr = "drop({}, {})".format(to_expr(s), ", ".join(identifiers))
    ret_typ = s.typ._drop(*identifiers)
    return convert_column(Column(expr, ret_typ))


@typecheck(x=numeric)
def exp(x):
    return _func("exp", TFloat64(), x)


@typecheck(c1=integral, c2=integral, c3=integral, c4=integral)
def fet(c1, c2, c3, c4):
    ret_typ = TStruct(['pValue', 'oddsRatio', 'ci95Lower', 'ci95Upper'],
                      [TFloat64(), TFloat64(), TFloat64(), TFloat64()])
    return _func("fet", ret_typ, c1, c2, c3, c4)


@typecheck(j=integral, k=integral)
def gt_index(j, k):
    return _func("gtIndex", TInt32(), j, k)


@typecheck(j=integral)
def gtj(j):
    return _func("gtj", TInt32(), j)


@typecheck(k=integral)
def gtk(k):
    return _func("gtk", TInt32(), k)


@typecheck(num_hom_ref=integral, num_het=integral, num_hom_var=integral)
def hwe(num_hom_ref, num_het, num_hom_var):
    ret_typ = TStruct(['rExpectedHetFrequency', 'pHWE'], [TFloat64(), TFloat64()])
    return _func("hwe", ret_typ, num_hom_ref, num_het, num_hom_var)


@typecheck(structs=oneof(ArrayStructColumn),
           identifier=string_typ)
def index(structs, identifier):  # FIXME: Need to be able to take list of Struct as input args
    struct_typ = get_typ(structs).element_type
    struct_fields = {fd.name: fd.typ for fd in struct_typ.fields}

    if identifier not in struct_fields:
        raise RuntimeError("`structs' does not have a field with identifier `{}'. " \
              "Struct type is {}.".format(identifier, struct_typ))

    key_typ = struct_fields[identifier]
    value_typ = struct_typ._drop(identifier)

    expr = "index({}, {})".format(to_expr(structs), identifier)
    ret_typ = TDict(key_typ, value_typ)
    return convert_column(Column(expr, ret_typ))


@typecheck(x=anytype)
def is_defined(x):
    return _func("isDefined", TBoolean(), x)


@typecheck(x=anytype)
def is_missing(x):
    return _func("isMissing", TBoolean(), x)


@typecheck(x=numeric)
def is_nan(x):
    return _func("isnan", TBoolean(), x)


@typecheck(x=anytype)
def json(x):
    return _func("json", TString(), x)


@typecheck(x=numeric, b=numeric)
def log(x, b=None):
    if b:
        return _func("log", TFloat64(), x, b)
    else:
        return _func("log", TFloat64(), x)


@typecheck(x=numeric)
def log10(x):
    return _func("log10", TFloat64(), x)


@typecheck(x=bool_typ)
def logical_not(x):
    return _func("!", TBoolean(), x)


@typecheck(s1=StructColumn, s2=StructColumn)  # FIXME: Need to be able to take regular Struct as input args
def merge(s1, s2):
    ret_typ = s1.typ._merge(s2.typ)
    return _func("merge", ret_typ, s1, s2)


@typecheck(a=anytype, b=anytype)
def or_else(a, b):
    return _func("orElse", get_typ(a), a, b)


@typecheck(a=anytype, b=anytype)
def or_missing(a, b):
    return _func("orMissing", get_typ(b), a, b)


@typecheck(x=numeric, df=numeric)
def pchisqtail(x, df):
    return _func("pchisqtail", TFloat64(), x, df)


@typecheck(p=numeric)
def pcoin(p):
    return _func("pcoin", TBoolean(), p)


@typecheck(x=numeric)
def pnorm(x):
    return _func("pnorm", TFloat64(), x)


@typecheck(b=numeric, x=numeric)
def pow(b, x):
    return _func("pow", TFloat64(), b, x)


@typecheck(x=numeric, lamb=numeric, lower_tail=bool_typ, logP=bool_typ)
def ppois(x, lamb, lower_tail=True, logP=False):
    return _func("ppois", TFloat64(), x, lamb, lower_tail, logP)


@typecheck(p=numeric, df=numeric)
def qchisqtail(p, df):
    return _func("qchisqtail", TFloat64(), p, df)


@typecheck(p=numeric)
def qnorm(p):
    return _func("qnorm", TFloat64(), p)


@typecheck(p=numeric, lamb=numeric, lower_tail=bool_typ, logP=bool_typ)
def qpois(p, lamb, lower_tail=True, logP=False):
    return _func("qpois", TInt32(), p, lamb, lower_tail, logP)


@typecheck(stop=integral, start=integral, step=integral)
def range(stop, start=0, step=1):
    return _func("range", TArray(TInt32()), start, stop, step)


@typecheck(mean=numeric, sd=numeric)
def rnorm(mean, sd):
    return _func("rnorm", TFloat64(), mean, sd)


@typecheck(lamb=numeric, n=integral)
def rpois(lamb, n=1):
    if n > 1:
        return _func("rpois", TArray(TFloat64()), n, lamb)
    else:
        return _func("rpois", TFloat64(), lamb)


@typecheck(min=numeric, max=numeric)
def runif(min, max):
    return _func("runif", TFloat64(), min, max)


@typecheck(s=StructColumn, identifiers=tupleof(string_typ))  # FIXME: Need to be able to take regular Struct as input args
def select(s, *identifiers):
    expr = "select({}, {})".format(to_expr(s), ", ".join(identifiers))
    ret_typ = s.typ._select(*identifiers)
    return convert_column(Column(expr, ret_typ))


@typecheck(x=numeric)
def sqrt(x):
    return _func("sqrt", TFloat64(), x)


@typecheck(x=anytype)
def to_str(x):
    return _func("str", TString(), x)


@typecheck(pred=bool_typ, then_case=anytype, else_case=anytype)
def where(pred, then_case, else_case):
    expr = "if ({}) {} else {}".format(to_expr(pred), to_expr(then_case), to_expr(else_case))
    ret_typ = get_typ(then_case)
    return convert_column(Column(expr, ret_typ))
