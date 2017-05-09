#!/usr/bin/env python

import os
import sys
import time
import signal
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str, help='Name of the cluster.')
parser.add_argument('--port', '-p', default='10000', type=str, help='Local port to use for SSH tunnel to master node.')
parser.add_argument('--zone', '-z', default='us-central1-b', type=str, help='Compute zone for Google cluster.')
args = parser.parse_args()

# check if Google Chrome is installed at default path
#chrome = os.path.abspath(r'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
#if not os.path.isfile(chrome):
#    print 'Connection failed - Google Chrome executable not found at "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".'
#    sys.exit()

# if process is currently running on local port designated for SSH tunnel, kill it
try:
    pid = subprocess.check_output(['lsof', '-t', '-i:{}'.format(args.port)]).split()
except subprocess.CalledProcessError:
    pass
else:
    for x in pid:
        os.kill(int(x), signal.SIGTERM)

# open SSH tunnel to master node
cmd = [
    'gcloud compute ssh {}-m'.format(args.name),
    '--zone={}'.format(args.zone),
    '--ssh-flag="-D {} -N -f -n"'.format(args.port),
    '> /dev/null 2>&1 &'
]
subprocess.call(' '.join(cmd), shell=True)

# wait for SSH tunnel to open
time.sleep(2)

# open Chrome with SOCKS proxy configuration
cmd = [
    r'/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome',
    'http://localhost:8123',
    '--proxy-server="socks5://localhost:{}"'.format(args.port),
    '--host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost"',
    '--user-data-dir=/tmp/',
    '> /dev/null 2>&1 &'
]
subprocess.call(' '.join(cmd), shell=True)
