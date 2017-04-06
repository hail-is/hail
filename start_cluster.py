#!/usr/bin/env python

import os
import time
import json
import argparse
from subprocess import call

parser = argparse.ArgumentParser()

# required arguments
parser.add_argument('--name', '-n', required=True, type=str, help='Name of cluster.')

# arguments with default parameters
parser.add_argument('--image-version', default='1.1', type=str, help='Google dataproc image version.')
parser.add_argument('--master-machine-type', '--master', '-m', default='n1-highmem-16', type=str, help='Master machine type.')
parser.add_argument('--master-boot-disk-size', default='100GB', type=str, help='Disk size of master machine.')
parser.add_argument('--metadata', default='', type=str, help='Metadata to be made available to the OS running on the instances.')
parser.add_argument('--num-master-local-ssds', default='0', type=str, help='Number of local SSDs to attach to the master machine.')
parser.add_argument('--num-preemptible-workers', '--n-pre-workers', '-np', default='0', type=str, help='Number of preemptible worker machines.')
parser.add_argument('--num-worker-local-ssds', default='0', type=str, help='Number of local SSDs to attach to each worker machine.')
parser.add_argument('--num-workers', '--n-workers', '-nw', default='2', type=str, help='Number of worker machines.')
parser.add_argument('--preemptible-worker-boot-disk-size', default='40GB', type=str, help='Disk size of preemptible machines.')
parser.add_argument('--worker-boot-disk-size', default='40GB', type=str, help='Disk size of worker machines.')
parser.add_argument('--worker-machine-type', '--worker', '-w', default='n1-standard-4', type=str, help='Worker machine type.')
parser.add_argument('--zone', default='us-central1-b', type=str, help='Compute zone for the cluster.')

# initialization action flags
parser.add_argument('--notebook', '-nb', action='store_true')
parser.add_argument('--vep', action='store_true')

# default Spark configuration properties
parser.add_argument('--spark-driver-memory', default='85g', type=str, help='Memory to allocate to Spark driver on master machine.')
parser.add_argument('--spark-driver-max-result-size', default='50g', type=str, help='Spark driver maxResultSize.')
parser.add_argument('--spark-task-max-failures', default='20', type=str, help='Maximum task failures allowed before job failure.')
parser.add_argument('--spark-kryo-buffer-max', default='1g', type=str, help='Kryoserializer buffer max.')

# default HDFS configuration properties
parser.add_argument('--hdfs-dfs-replication', default='1', type=str, help='HDFS DFS replications.')

# parse arguments
args = parser.parse_args()

# parse Spark and HDFS configuration parameters, combine into properties argument
spark_properties = [
    'spark:spark.driver.memory={}'.format(args.spark_driver_memory),
    'spark:spark.driver.maxresultSize={}'.format(args.spark_driver_max_result_size),
    'spark:spark.task.maxFailures={}'.format(args.spark_task_max_failures),
    'spark:spark.kryoserializer.buffer.max={}'.format(args.spark_kryo_buffer_max),
    'spark:spark.driver.extraJavaOptions=-Xss4M',
    'spark:spark.executor.extraJavaOptions=-Xss4M'
]
hdfs_properties = [
    'hdfs:dfs.replication={}'.format(args.hdfs_dfs_replication)
]
properties = ','.join(spark_properties + hdfs_properties)

# parse metadata key/values
if args.metadata:
    ## TO DO
    pass
    pass_metadata = ''
else:
    pass_metadata = ''

# default initialization script to start up cluster with
init_actions = 'gs://hail-common/init_default.py'

# add notebook action
if args.notebook:
    init_actions = init_actions + ',' + 'gs://hail-common/init_notebook.py'

# add VEP action
if args.vep:
    init_actions = init_actions + ',' + 'gs://hail-common/vep/vep/vep85-init.sh'
    
# command to start cluster
cmd = ' '.join([
    'gcloud dataproc clusters create',
    args.name,
    '--image-version={}'.format(args.image_version),
    '--master-machine-type={}'.format(args.master_machine_type),
    '--master-boot-disk-size={}'.format(args.master_boot_disk_size),
    '{}'.format(pass_metadata),
    '--num-master-local-ssds={}'.format(args.num_master_local_ssds),
    '--num-preemptible-workers={}'.format(args.num_preemptible_workers),
    '--num-worker-local-ssds={}'.format(args.num_worker_local_ssds),
    '--num-workers={}'.format(args.num_workers),
    '--preemptible-worker-boot-disk-size={}'.format(args.preemptible_worker_boot_disk_size),
    '--worker-boot-disk-size={}'.format(args.worker_boot_disk_size),
    '--worker-machine-type={}'.format(args.worker_machine_type),
    '--zone={}'.format(args.zone),
    '--properties={}'.format(properties),
    '--initialization-actions={}'.format(init_actions)
])

# spin up cluster
call(cmd, shell=True)

# wait for Jupyter server process to start if notebook action is taken
if args.notebook:
    print "Waiting for Jupyter notebook server to start..."
    time.sleep(30)
    print "Done!"
