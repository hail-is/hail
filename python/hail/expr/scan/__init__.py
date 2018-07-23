from ...expr import aggregators
from functools import wraps, update_wrapper


def scan_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        func = getattr(f, '__wrapped__')
        af = func.__globals__['_agg_func']
        setattr(af, 'as_scan', True)
        res = f(*args, **kwargs)
        setattr(af, 'as_scan', False)
        return res
    update_wrapper(wrapper, f)
    return wrapper


__all__ = [name for name in dir(aggregators) if name[0] != '_']


for name in __all__:
    globals()[name] = scan_decorator(getattr(aggregators, name))

del scan_decorator, name, aggregators
