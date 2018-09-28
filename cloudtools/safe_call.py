import subprocess as sp
import sys

if sys.version_info >= (3,0):
    def decode(x):
        return x.decode()
else:
    def decode(x):
        return x

def safe_call(*args):
    '''only print output on error'''
    try:
        sp.check_output(args, stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print(decode(e.output))
        raise e
