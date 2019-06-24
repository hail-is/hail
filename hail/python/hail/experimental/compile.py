import tempfile
import ctypes

from ..utils.java import Env
from .codec import encode
from ..expr.functions import literal


def compile_comparison_binary(op, codecName, l_type, r_type):
    return Env.hc()._jhc.backend().compileComparisonBinary(
        op, codecName, l_type, r_type)


def compiled_compare(l, r, codec='unblockedUncompressed'):
    type1, arr1 = encode(literal(l), codec)
    arr1size = len(arr1)
    type2, arr2 = encode(literal(r), codec)
    arr2size = len(arr1)
    print(f'{type1} {arr1} {arr1size} {type2} {arr2} {arr2size}')
    # FIXME: do this once
    ctypes.CDLL('/Users/dking/projects/hail/hail/prebuilt/lib/darwin/libhail.dylib',
                mode=ctypes.RTLD_GLOBAL)
    compare_binary = compile_comparison_binary("compare", codec, type1, type2)
    with tempfile.NamedTemporaryFile() as f:
        f.write(compare_binary)
        lib = ctypes.CDLL(f.name)
        return lib.compare(arr1, arr1size, arr2, arr2size)
