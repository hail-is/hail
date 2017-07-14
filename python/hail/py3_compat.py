import sys

if sys.version_info > (3,):
    xrange = range
    long = int
    unicode = str

    def iteritems(x):
        return x.items()

    def map_list(*args, **kwargs):
        return list(map(*args, **kwargs))
else:
    def iteritems(x):
        return x.iteritems()

    map_list = map