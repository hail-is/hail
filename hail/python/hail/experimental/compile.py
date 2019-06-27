import tempfile
import ctypes

from ..utils.java import Env
from .codec import encode
from ..expr.functions import literal

__libhail_loaded = False


def load_libhail():
    ctypes.CDLL('/Users/dking/projects/hail/hail/src/main/c/lib/darwin/libhail.dylib',
                mode=ctypes.RTLD_GLOBAL)
    global __libhail_loaded
    __libhail_loaded = True


def compile_comparison_binary(op, codecName, l_type, r_type):
    return Env.hc()._jhc.backend().compileComparisonBinary(
        op, codecName, l_type, r_type)


def compiled_compare(l, r, codec='unblockedUncompressed'):
    type1, arr1 = encode(literal(l), codec)
    arr1size = len(arr1)
    type2, arr2 = encode(literal(r), codec)
    arr2size = len(arr1)
    load_libhail()
    compare_binary = compile_comparison_binary("compare", codec, type1, type2)
    with tempfile.NamedTemporaryFile() as f:
        f.write(compare_binary)
        lib = ctypes.CDLL(f.name)
        return lib.compare(arr1, arr1size, arr2, arr2size)
