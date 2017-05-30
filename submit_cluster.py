#!/usr/bin/env python

import argparse
from subprocess import call, Popen, PIPE

parser = argparse.ArgumentParser()
parser.add_argument('--name', '-n', required=True, type=str)
parser.add_argument('--hash', required=False, type=str)
parser.add_argument('--jar', required=False, type=str)
parser.add_argument('--zip', required=False, type=str)
parser.add_argument('--properties', '-p', required=False, type=str)
parser.add_argument('script', type=str)
args = parser.parse_args()

# get Hail hash using either most recent, or an older version if specified
if args.hash:
    hail_hash = args.hash
else:
    hail_hash = Popen(['gsutil', 'cat', 'gs://hail-common/latest-hash.txt'], stdout=PIPE, stderr=PIPE).communicate()[0].strip()

# Hail jar
if args.jar:
    hail_jar = args.jar.rsplit('/')[-1]
    jar_path = args.jar
else:
    hail_jar = 'hail-hail-is-master-all-2.0.2-{}.jar'.format(hail_hash)
    jar_path = 'gs://hail-common/' + hail_jar

# Hail archive
if args.zip:
    hail_zip = args.zip.rsplit('/')[-1]
    zip_path = args.zip
else:
    hail_zip = 'pyhail-hail-is-master-{}.zip'.format(hail_hash)
    zip_path = 'gs://hail-common/' + hail_zip

# create properties argument
properties = 'spark.driver.extraClassPath=./{0},spark.executor.extraClassPath=./{0}'.format(hail_jar)
if args.properties:
    properties = properties + ',' + args.properties

# pyspark submit command
cmd = ' '.join([
    'gcloud',
    'dataproc',
    'jobs',
    'submit',
    'pyspark',
    args.script,
    '--cluster={}'.format(args.name),
    '--files={}'.format(jar_path),
    '--py-files={}'.format(zip_path),
    '--properties={}'.format(properties)
])
call(cmd, shell=True)
