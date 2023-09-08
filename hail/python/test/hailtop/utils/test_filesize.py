from pytest import raises
from hailtop.utils.filesize import filesize

def test_filesize():
    for n in [-1, -1023, -1024, -1025, -1024**3]:
        with raises(ValueError):
            filesize(n)
    assert filesize(0) == "0B"
    assert filesize(1) == "1B"
    assert filesize(2) == "2B"
    assert filesize(1023) == "1023B"
    assert filesize(1024) == "1KiB"
    assert filesize(1025) == "1KiB"
    prefixes = ["K", "M", "G", "T", "P", "E", "Z", "Y"]
    for exp in range(2, 9):
        assert filesize(1024 ** exp - 1) == f"1023{prefixes[exp - 2]}iB"
        assert filesize(1024 ** exp) == f"1{prefixes[exp - 1]}iB"
        assert filesize(1024 ** exp + 1) == f"1{prefixes[exp - 1]}iB"
        assert filesize(1024 ** exp * 2) == f"2{prefixes[exp - 1]}iB"
    assert filesize(1024 ** 9 - 1) == "1023YiB"
    assert filesize(1024 ** 9) == "1024YiB"
    assert filesize(1024 ** 9 + 1) == "1024YiB"
    assert filesize(1024 ** 9 * 2) == "2048YiB"
    assert filesize(1024 ** 10) == "1048576YiB"

