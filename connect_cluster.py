#!/usr/bin/env python

import time
import argparse
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str, help='Name of cluster.')
parser.add_argument('--notebook', '--nb', action='store_true')
args = parser.parse_args()

# open SSH tunnel to master machine
cmd = ' '.join([
    'gcloud compute ssh',
    '{}'.format(args.name + '-m'),
    '--ssh-flag="-D 10000 -N -f -n"',
    '> /dev/null 2>&1 &'
])
call(cmd, shell=True)

# if notebook flag, open connection to 8123, otherwise open to 4040 (Spark UI)
if args.notebook:
    port = '8123'
else:
    port = '4040'

# wait
time.sleep(2)

# open Chrome with SOCKS proxy configuration
browser_exec = r'/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome'
cmd = ' '.join([
    browser_exec,
    'http://localhost:{}'.format(port),
    '--proxy-server="socks5://localhost:10000"',
    '--host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost"',
    '--user-data-dir=/tmp/',
    '> /dev/null 2>&1 &'
])
call(cmd, shell=True)
