#!/usr/bin/env python

import argparse
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str)
args = parser.parse_args()

call(['gcloud', 'dataproc', 'clusters', 'delete', args.name])
