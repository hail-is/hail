import re

from hailtop.utils import secret_alnum_string

_GCE_MAX_VM_NAME_LEN = 63


def make_machine_name(prefix: str) -> str:
    # Suffix has no hyphens so the last '-' in the full name is always the inst_coll separator.
    # Fill remaining space up to 12 chars (36^12 ≈ 4.7e18 unique values at full length).
    available = _GCE_MAX_VM_NAME_LEN - len(prefix)
    assert available > 0, f'machine name prefix is too long for GCE: {prefix!r}'
    suffix_len = min(12, available)
    return f'{prefix}{secret_alnum_string(suffix_len, case="lower")}'


def build_inst_coll_regex(machine_name_prefix: str) -> re.Pattern:
    # The suffix contains no hyphens, so greedy .* naturally stops at the last '-'.
    # This correctly handles hyphenated inst_coll names (e.g. 'standard-np').
    return re.compile(f'{machine_name_prefix}(?P<inst_coll>.*)-[a-z0-9]+$')
