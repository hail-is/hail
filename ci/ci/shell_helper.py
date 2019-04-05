import subprocess as sp


def shell(*args):
    sp.run(args, stderr=sp.PIPE, stdout=sp.PIPE, check=True)
