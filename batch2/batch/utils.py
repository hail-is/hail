import re
import secrets
import logging

log = logging.getLogger('utils')


def new_token(n=5):
    return ''.join([secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(n)])


cpu_regex = re.compile(r"^(\d*\.\d+|\d+)([m]?)$")


def parse_cpu_in_mcpu(cpu_string):
    match = cpu_regex.fullmatch(cpu_string)
    if match:
        number = float(match.group(1))
        if match.group(2) == 'm':
            number /= 1000
        return int(number * 1000)
    return None


image_regex = re.compile(r"(?:.+/)?([^:]+)(:(.+))?")


def parse_image_tag(image_string):
    match = image_regex.fullmatch(image_string)
    if match:
        return match.group(3)
    return None
