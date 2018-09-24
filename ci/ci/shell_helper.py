import subprocess as sp


def shell(*args):
    sp.run(args, capture_output=True, check=True)
