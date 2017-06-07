from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.expr.column import Column, args_to_expr, convert_column, get_typ
from hail.types import *
from hail.java import *
from hail.typecheck import *


def _func(name, ret_typ, *args):
    @args_to_expr
    def _(*args):
        return convert_column(Column("{name}({args})".format(name=name, args=", ".join(args)), ret_typ))
    return _(*args)


def chisq(c1, c2, c3, c4):
    ret_typ = TStruct(['pValue', 'oddsRatio'], [TDouble(), TDouble()])
    return _func("chisq", ret_typ, c1, c2, c3, c4)

def combine_variants(left, right):
    ret_typ = TStruct(['variant', 'laIndices', 'raIndices'], [TVariant(), TDict(TInt(), TInt()), TDict(TInt(), TInt())])
    return _func("combineVariants", ret_typ, left, right)

def ctt(c1, c2, c3, c4, min_cell_count):
    ret_typ = TStruct(['pValue', 'oddsRatio'], [TDouble(), TDouble()])
    return _func("ctt", ret_typ, c1, c2, c3, c4, min_cell_count)

def Dict(keys, values):
    key_typ = get_typ(keys)
    value_typ = get_typ(values)
    ret_typ = TDict(key_typ, value_typ)
    return _func("Dict", ret_typ, keys, values)

def dpois(x, lamb, logP=False):
    return _func("dpois", TDouble(), x, lamb, logP)

def drop(s, *identifiers):
    ret_typ = s.typ._drop(*identifiers)
    return _func("drop", ret_typ, s, *identifiers)

def exp(x):
    return _func("exp", TDouble(), x)

def fet(c1, c2, c3, c4):
    ret_typ = TStruct(['pValue', 'oddsRatio', 'ci95Lower', 'ci95Upper'],
                      [TDouble(), TDouble(), TDouble(), TDouble()])
    return _func("fet", ret_typ, c1, c2, c3, c4)

def Genotype(v, call, ad, dp, gq, pl):
    return _func("Genotype", TGenotype(), v, call, ad, dp, gq, pl)

### FIXME: Genotype has multiple signatures

def gt_index(j, k):
    return _func("gtIndex", TInt(), j, k)

def gtj(j):
    return _func("gtj", TInt(), j)

def gtk(k):
    return _func("gtk", TInt(), k)

def hwe(num_hom_ref, num_het, num_hom_var):
    ret_typ = TStruct(['rExpectedHetFrequency', 'pHWE'], [TDouble(), TDouble()])
    return _func("hwe", ret_typ, num_hom_ref, num_het, num_hom_var)

def index(structs, identifier):
    # if structs:
    #     struct_ret_typ = structs[0]
    # ret_typ = TDict(TString(), )
    pass ### this return type is hard

def Interval(chr, start, end):
    return _func("Interval", TInterval(), chr, start, end)

### Interval has multiple signatures

def is_defined(x):
    return _func("isDefined", TBoolean(), x)

def is_missing(x):
    return _func("isMissing", TBoolean(), x)

def json(x):
    return _func("json", TString(), x)

def Locus(contig, pos):
    return _func("Locus", TLocus(), contig, pos)

### FIXME: Locus has multiple signatures

def log(x, b):
    return _func("log", TDouble(), x, b)

### FIXME: multiple signatures

def log10(x):
    return _func("log10", TDouble(), x)

def logical_not(x):
    return _func("!", TBoolean(), x)

def merge(s1, s2):
    ret_typ = s1.typ._merge(s2.typ)
    return _func("merge", ret_typ, s1, s2)

def or_else(a, b):
    return _func("orElse", get_typ(a), a, b)

def or_missing(a, b):
    return _func("orMissing", get_typ(b), a, b)

def pchisqtail(x, df):
    return _func("pchisqtail", TDouble(), x, df)

def pcoin(p):
    return _func("pcoin", TBoolean(), p)

def pnorm(x):
    return _func("pnorm", TDouble(), x)

def pow(b, x):
    return _func("pow", TDouble(), b, x)

def ppois(x, lamb, lower_tail=True, logP=False):
    return _func("ppois", TDouble(), x, lamb, lower_tail, logP)

def qchisqtail(p, df):
    return _func("qchisqtail", TDouble(), p, df)

def qnorm(p):
    return _func("qnorm", TDouble(), p)

def qpois(p, lamb, lower_tail=True, logP=False):
    return _func("qpois", TInt(), p, lamb, lower_tail, logP)

def range(stop, start=0, step=1):
    return _func("range", TArray(TInt()), start, stop, step)

def rnorm(mean, sd):
    return _func("rnorm", TDouble(), mean, sd)

def rpois(lamb, n=1):
    if n > 1:
        return _func("rpois", TArray(TDouble()), n, lamb)
    else:
        return _func("rpois", TDouble(), lamb)

def runif(min, max):
    return _func("runif", TDouble(), min, max)

def select(s, *identifiers):
    pass

def sqrt(x):
    return _func("sqrt", TDouble(), x)

### FIXME: str

#### Variant has multiple signatures
#

### FIXME: return type of where???
# def where(x, y, z):
#     @args_to_expr
#     def _(x, y, z):
#         return convert_column(Column("if ({x}) {y} else {z}".format(x=x, y=y, z=z), ret_typ))
#     return _(x, y, z)


