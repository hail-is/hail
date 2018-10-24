from .utils import decode
import subprocess as sp
import sys

def safe_call(*args):
    '''only print output on error'''
    try:
        sp.check_output(args, stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print(decode(e.output))
        raise e
