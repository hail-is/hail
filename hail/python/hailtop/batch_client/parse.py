from typing import Optional, Mapping, Pattern
import re
import math

MEMORY_REGEXPAT: str = r'[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?'
MEMORY_REGEX: Pattern = re.compile(MEMORY_REGEXPAT)

CPU_REGEXPAT: str = r'[+]?((?:[0-9]*[.])?[0-9]+)([m])?'
CPU_REGEX: Pattern = re.compile(CPU_REGEXPAT)

STORAGE_REGEXPAT: str = r'[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?'
STORAGE_REGEX: Pattern = re.compile(STORAGE_REGEXPAT)


def parse_cpu_in_mcpu(cpu_string: str) -> Optional[int]:
    match = CPU_REGEX.fullmatch(cpu_string)
    if match:
        number = float(match.group(1))
        if match.group(2) == 'm':
            number /= 1000
        return int(number * 1000)
    return None


conv_factor: Mapping[str, int] = {
    'K': 1000, 'Ki': 1024,
    'M': 1000**2, 'Mi': 1024**2,
    'G': 1000**3, 'Gi': 1024**3,
    'T': 1000**4, 'Ti': 1024**4,
    'P': 1000**5, 'Pi': 1024**5
}


def parse_memory_in_bytes(memory_string: str) -> Optional[int]:
    match = MEMORY_REGEX.fullmatch(memory_string)
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix:
            return math.ceil(number * conv_factor[suffix])
        return math.ceil(number)
    return None


def parse_storage_in_bytes(storage_string: str) -> Optional[int]:
    match = STORAGE_REGEX.fullmatch(storage_string)
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix:
            return math.ceil(number * conv_factor[suffix])
        return math.ceil(number)
    return None
