from ...expr import aggregators
from functools import wraps, update_wrapper
import sys

def scan_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        func = getattr(f, '__wrapped__')
        af = func.__globals__['_agg_func']
        setattr(af, '_as_scan', True)
        res = f(*args, **kwargs)
        setattr(af, '_as_scan', False)
        return res
    update_wrapper(wrapper, f)
    return wrapper


__all__ = [name for name in dir(aggregators) if name[0] != '_']


thismodule = sys.modules[__name__]
for name in __all__:
    setattr(thismodule, name, scan_decorator(getattr(aggregators, name)))

del scan_decorator, name, aggregators, sys, thismodule
