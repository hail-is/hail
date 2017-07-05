#!/usr/bin/env python

import os
import time
import json
import argparse
import subprocess

# master machine type to memory map, used for setting spark.driver.memory property
machine_mem = {
    'n1-standard-1': 3.75,
    'n1-standard-2': 7.5,
    'n1-standard-4': 15,
    'n1-standard-8': 30,
    'n1-standard-16': 60,
    'n1-standard-32': 120,
    'n1-standard-64': 240,
    'n1-highmem-2': 13,
    'n1-highmem-4': 26,
    'n1-highmem-8': 52,
    'n1-highmem-16': 104,
    'n1-highmem-32': 208,
    'n1-highmem-64': 416,
    'n1-highcpu-2': 1.8,
    'n1-highcpu-4': 3.6,
    'n1-highcpu-8': 7.2,
    'n1-highcpu-16': 14.4,
    'n1-highcpu-32': 28.8,
    'n1-highcpu-64': 57.6
}


def main(args):
    # Google dataproc image version to use
    if args.spark == '2.0.2':
        image_version = '1.1'
    elif args.spark == '2.1.0':
        image_version = 'preview'

    # default to highmem machines if using VEP
    if not args.worker_machine_type:
        if args.vep:
            args.worker_machine_type = 'n1-highmem-8'
        else:
            args.worker_machine_type = 'n1-standard-8'  # default
    
    # parse Spark and HDFS configuration parameters, combine into properties argument
    properties = [
        'spark:spark.driver.memory={}g'.format(str(int(machine_mem[args.master_machine_type] * 0.8))),
        'spark:spark.driver.maxResultSize=0',
        'spark:spark.task.maxFailures=20',
        'spark:spark.kryoserializer.buffer.max=1g',
        'spark:spark.driver.extraJavaOptions=-Xss4M',
        'spark:spark.executor.extraJavaOptions=-Xss4M',
        'hdfs:dfs.replication=1'
    ]
    if args.properties:
        properties.append(args.properties)

    # default initialization script to start up cluster with
    init_actions = args.init

    # add VEP action
    if args.vep:
        init_actions += ',' + 'gs://hail-common/vep/vep/vep85-init.sh'

    if args.hash == 'latest':
        hail_hash = subprocess.Popen(['gsutil', 'cat', 'gs://hail-common/latest-hash.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].strip()
    else:
        hail_hash = args.hash

    # prepare metadata values
    metadata = 'HASH={0},SPARK={1}'.format(hail_hash, args.spark)

    # if Hail jar and zip, add to metadata
    if args.jar:
        metadata += ',JAR={}'.format(args.jar)
    if args.zip:
        metadata += ',ZIP={}'.format(args.zip)

    # command to start cluster
    cmd = ' '.join([
        'gcloud dataproc clusters create',
        args.name,
        '--image-version={}'.format(image_version),
        '--master-machine-type={}'.format(args.master_machine_type),
        '--metadata {}'.format(metadata),
        '--master-boot-disk-size={}GB'.format(args.master_boot_disk_size),
        '--num-master-local-ssds={}'.format(args.num_master_local_ssds),
        '--num-preemptible-workers={}'.format(args.num_preemptible_workers),
        '--num-worker-local-ssds={}'.format(args.num_worker_local_ssds),
        '--num-workers={}'.format(args.num_workers),
        '--preemptible-worker-boot-disk-size={}GB'.format(args.preemptible_worker_boot_disk_size),
        '--worker-boot-disk-size={}GB'.format(args.worker_boot_disk_size),
        '--worker-machine-type={}'.format(args.worker_machine_type),
        '--zone={}'.format(args.zone),
        '--properties={}'.format(",".join(properties)),
        '--initialization-actions={}'.format(init_actions)
    ])

    # spin up cluster
    try:
        subprocess.check_output(cmd, shell=True)

    # exit if cluster already exists
    except subprocess.CalledProcessError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--name', '-n', required=True, type=str, help='Name of cluster.')

    # arguments with default parameters
    parser.add_argument('--hash', default='latest', type=str, help='Hail build to use for notebook initialization.')
    parser.add_argument('--spark', default='2.0.2', type=str, choices=['2.0.2', '2.1.0'], help='Spark version used to build Hail.')
    parser.add_argument('--master-machine-type', '--master', '-m', default='n1-highmem-8', type=str, help='Master machine type.')
    parser.add_argument('--master-boot-disk-size', default=100, type=int, help='Disk size of master machine (in GB).')
    parser.add_argument('--num-master-local-ssds', default=0, type=int, help='Number of local SSDs to attach to the master machine.')
    parser.add_argument('--num-preemptible-workers', '--n-pre-workers', '-np', default=0, type=int, help='Number of preemptible worker machines.')
    parser.add_argument('--num-worker-local-ssds', default=0, type=int, help='Number of local SSDs to attach to each worker machine.')
    parser.add_argument('--num-workers', '--n-workers', '-nw', default=2, type=int, help='Number of worker machines.')
    parser.add_argument('--preemptible-worker-boot-disk-size', default=40, type=int, help='Disk size of preemptible machines (in GB).')
    parser.add_argument('--worker-boot-disk-size', default=40, type=int, help='Disk size of worker machines (in GB).')
    parser.add_argument('--worker-machine-type', '--worker', '-w', type=str, help='Worker machine type.')
    parser.add_argument('--zone', default='us-central1-b', type=str, help='Compute zone for the cluster.')
    parser.add_argument('--properties', default='', type=str, help='Additional configuration properties for the cluster.')

    # specify custom Hail jar and zip
    parser.add_argument('--jar', default='', type=str, help='Hail jar to use for Jupyter notebook.')
    parser.add_argument('--zip', default='', type=str, help='Hail zip to use for Jupyter notebook.')

    # initialization action flags
    parser.add_argument('--init', default='gs://hail-common/init_notebook.py', help='comma-separated list of init scripts to run.')
    parser.add_argument('--vep', action='store_true')

    # parse arguments
    args = parser.parse_args()

    main(args)
