import ctypes
import pkg_resources
import platform
import tempfile
import zipfile

from ..utils.java import Env
from .codec import encode
from ..expr.functions import literal

__libhail_loaded = False


def load_libhail():
    global __libhail_loaded
    if __libhail_loaded:
        return
    with zipfile.ZipFile(pkg_resources.resource_stream(
            'hail',
            "hail-all-spark.jar")) as jar:
        if platform.system() == 'Darwin':
            lib_path = 'darwin/libhail.dylib'
        else:
            lib_path = 'linux-x86-64/libhail.so'
        with tempfile.NamedTemporaryFile() as f:
            f.write(jar.read(lib_path))
            ctypes.CDLL(f.name, mode=ctypes.RTLD_GLOBAL)
    __libhail_loaded = True


def compile_comparison_binary(op, codecName, l_type, r_type):
    return Env.backend()._jhc.backend().compileComparisonBinary(
        op, codecName, l_type, r_type)


def compiled_compare(left, right, codec='unblockedUncompressed'):
    type1, arr1 = encode(literal(left), codec)
    arr1size = len(arr1)
    type2, arr2 = encode(literal(right), codec)
    arr2size = len(arr1)
    load_libhail()
    compare_binary = compile_comparison_binary("compare", codec, type1, type2)
    with tempfile.NamedTemporaryFile() as f:
        f.write(compare_binary)
        lib = ctypes.CDLL(f.name)
        return lib.compare(arr1, arr1size, arr2, arr2size)
