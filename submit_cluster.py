#!/usr/bin/env python

import argparse
from subprocess import call, Popen, PIPE

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str)
parser.add_argument('--hash', required=False, type=str)
parser.add_argument('script', type=str)
args = parser.parse_args()

# get Hail hash using either most recent, or an older version if specified
if args.hash:
    hail_hash = args.hash
else:
    hail_hash = Popen('gsutil cat gs://hail-common/latest-hash.txt', shell=True, stdout=PIPE, stderr=PIPE).communicate()[0].strip()

# Hail files
hail_jar = 'hail-hail-is-master-all-spark2.0.2-{}.jar'.format(hail_hash)
hail_zip = 'pyhail-hail-is-master-{}.zip'.format(hail_hash)

# pyspark submit command
cmd = ' '.join([
    'gcloud dataproc jobs submit pyspark',
    args.script,
    '--cluster={}'.format(args.cluster),
    '--files={}'.format('gs://hail-common/' + hail_jar),
    '--py-files={}'.format('gs://hail-common/' + hail_zip),
    '--properties="spark.driver.extraClassPath=./{0},spark.executor.extraClassPath=./{0}"'.format(hail_jar)
])
call(cmd, shell=True)
