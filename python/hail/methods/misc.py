from decorator import decorator
from hail.api2 import MatrixTable
from hail.utils.java import Env

@decorator
def require_biallelic(f, dataset, *args, **kwargs):
    dataset = MatrixTable(Env.hc(), Env.hail().methods.VerifyBiallelic.apply(dataset._jvds, f.__name__))
    return f(dataset, *args, **kwargs)
