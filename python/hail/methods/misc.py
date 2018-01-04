from decorator import decorator
from hail.api2 import MatrixTable
from hail.utils.java import Env

@decorator
def require_biallelic(f, dataset, *args, **kwargs):
    from hail.expr.types import TVariant
    if not isinstance(dataset.rowkey_schema, TVariant):
        raise TypeError("Method '{}' requires the row key to be of type 'TVariant', found '{}'".format(
            f.__name__, dataset.rowkey_schema))
    dataset = MatrixTable(Env.hail().methods.VerifyBiallelic.apply(dataset._jvds, f.__name__))
    return f(dataset, *args, **kwargs)
