import subprocess as sp


def safe_call(*args):
    # only print output on error
    try:
        sp.check_output(args, stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print(e.output.decode())
        raise e
