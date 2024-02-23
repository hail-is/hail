import sys

try:
    import uvloop

    def install():
        return uvloop.install()
except ImportError as e:
    if not sys.platform.startswith('win32'):
        raise e

    def install():
        pass
