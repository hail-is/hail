import subprocess as sp


def flatten(xxs):
    return [x for xs in xxs for x in xs]


def shell(*args):
    sp.run(args, stderr=sp.PIPE, stdout=sp.PIPE, check=True)
