import hail.expr.functions as functions
import hail.expr.aggregators as agg
import hail.methods as methods
from hail.expr import Type, TInt32, TInt64, TFloat32, TFloat64, TString, TBoolean, TArray, TSet, TDict, TStruct, \
    TLocus, TVariant, TAltAllele, TCall, TInterval, eval_expr, eval_expr_typed
from hail.genetics import LDMatrix, KinshipMatrix, Variant, Locus, AltAllele, Interval, Call, Pedigree, Trio, \
    GenomeReference
from hail.utils import Struct, hadoop_write, hadoop_read, hadoop_copy
from hail.api2 import MatrixTable, Table, asc, desc
from hail.typecheck import nullable, strlike, integral, typecheck
from hail.utils.java import Env
from pyspark import SparkContext

def stop():
    Env.hc().stop()

@typecheck(sc=nullable(SparkContext),
                  app_name=strlike,
                  master=nullable(strlike),
                  local=strlike,
                  log=strlike,
                  quiet=bool,
                  append=bool,
                  min_block_size=integral,
                  branching_factor=integral,
                  tmp_dir=strlike,
                  default_reference=strlike)
def init(sc=None, app_name="Hail", master=None, local='local[*]',
             log='hail.log', quiet=False, append=False,
             min_block_size=1, branching_factor=50, tmp_dir='/tmp',
             default_reference="GRCh37"):
    from hail.api1 import HailContext
    HailContext(sc, app_name, master, local, log, quiet, append, min_block_size, branching_factor, tmp_dir, default_reference)

def default_reference():
    return Env.hc().default_reference

__all__ = ['Table',
           'MatrixTable',
           'asc',
           'desc',
           'Variant',
           'Locus',
           'AltAllele',
           'Interval',
           'Struct',
           'Call',
           'Pedigree',
           'Trio',
           'GenomeReference',
           'Type',
           'TInt32',
           'TInt64',
           'TFloat32',
           'TFloat64',
           'TString',
           'TBoolean',
           'TArray',
           'TSet',
           'TDict',
           'TStruct',
           'TLocus',
           'TVariant',
           'TAltAllele',
           'TCall',
           'TInterval',
           'hadoop_read',
           'hadoop_write',
           'hadoop_copy',
           'KinshipMatrix',
           'LDMatrix',
           'functions',
           'agg',
           'methods',
           'eval_expr',
           'eval_expr_typed',
           'default_reference',
           'init',
           'stop']
