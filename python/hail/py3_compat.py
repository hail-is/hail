import sys

if sys.version_info > (3,):
    xrange = range
    long = int
    unicode = str

    def iteritems(x):
        return x.items()
else:
    def iteritems(x):
        return x.iteritems()
