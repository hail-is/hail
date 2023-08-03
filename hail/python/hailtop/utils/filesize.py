_PREFIXES = ["B", "K", "M", "G", "T", "P", "E", "Z", "Y"]


def filesize(size: int) -> str:
    """
    Renders the given number as a human-readable filesize using binary multiple-byte units (e.g. "MiB", not "MB").
    Rounds down to the nearest integer value and multiple-byte unit (e.g. ``filesize((1024**3) - 1)`` returns
    ``"1023MiB"``, not ``"1GiB"`` or ``"1023.9999990463257MiB"``).
    """
    if size < 0:
        raise ValueError(f"Expected filesize to be non-negative, but received a value of {size} bytes.")
    idx = 0
    while size >= 1024 and idx < len(_PREFIXES) - 1:
        size //= 1024
        idx += 1
    suffix = "" if idx == 0 else "iB"
    return f"{size}{_PREFIXES[idx]}{suffix}"
