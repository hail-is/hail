import sys

try:
    import uvloop

    def install():  # pyright: ignore[reportRedeclaration]
        return uvloop.install()

except ImportError as e:
    if not sys.platform.startswith('win32'):
        raise e

    def install():  # pyright: ignore[reportRedeclaration]
        pass
