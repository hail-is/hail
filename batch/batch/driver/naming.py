import re

from hailtop.utils import secret_alnum_string


def make_machine_name(prefix: str) -> str:
    # 36 ** 12 = ~4.7e18 unique suffixes
    suffix = f'{secret_alnum_string(6, case="lower")}-{secret_alnum_string(6, case="lower")}'
    return f'{prefix}{suffix}'


def build_inst_coll_regex(machine_name_prefix: str) -> re.Pattern:
    # The suffix must be length-specific so the greedy .* doesn't consume hyphenated
    # inst_coll names (e.g. 'standard-np') into the inst_coll group.
    # Old suffix: 5 alphanum chars. New suffix: two 6-char alphanum groups separated by a hyphen.
    return re.compile(f'{machine_name_prefix}(?P<inst_coll>.*)-(?:[a-z0-9]{{5}}|[a-z0-9]{{6}}-[a-z0-9]{{6}})$')
