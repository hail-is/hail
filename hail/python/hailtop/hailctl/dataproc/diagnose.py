import re
import json
from subprocess import call, Popen, PIPE
import click

from .dataproc import dataproc


@dataproc.command(
    help="Diagnose problems in a Dataproc cluster.")
@click.argument('cluster_name')
@click.option('--dest', '-d',
              required=True,
              help="Directory for diagnose output -- must be local.")
@click.option('--hail-log', '-l',
              default='/home/hail/hail.log', show_default=True,
              help="Path for hail.log file.")
@click.option('--overwrite', is_flag=True,
              help="Delete dest directory before adding new files.")
@click.option('--no-diagnose', is_flag=True,
              help="Do not run gcloud dataproc clusters diagnose.")
@click.option('--compress', '-z', is_flag=True,
              help="GZIP all files.")
@click.option('--workers', multiple=True,
              help="Specific workers to get log files from.")
@click.option('--take',
              type=int, default=None,
              help="Only download logs from the first N workers.")
def diagnose(cluster_name, dest, hail_log, overwrite, no_diagnose, compress, workers, take):
    print("Diagnosing cluster '{}'...".format(cluster_name))

    is_local = not dest.startswith("gs://")

    if overwrite:
        if is_local:
            call('rm -r {dir}'.format(dir=dest), shell=True)
        else:
            call('gsutil -m rm -r {dir}'.format(dir=dest), shell=True)

    master_dest = dest.rstrip('/') + "/master/"
    worker_dest = dest.rstrip('/') + "/workers/"

    if is_local:
        call('mkdir -p {dir}'.format(dir=master_dest), shell=True)
        call('mkdir -p {dir}'.format(dir=worker_dest), shell=True)

    desc = json.loads(Popen('gcloud dataproc clusters describe {name} --format json'.format(name=cluster_name),
                            shell=True,
                            stdout=PIPE,
                            stderr=PIPE).communicate()[0].strip())

    config = desc['config']

    master = config['masterConfig']['instanceNames'][0]
    try:
        actual_workers = config['workerConfig']['instanceNames'] + config['secondaryWorkerConfig']['instanceNames']
    except KeyError:
        actual_workers = config['workerConfig']['instanceNames']
    zone = re.search(r'zones/(?P<zone>\S+)$', config['gceClusterConfig']['zoneUri']).group('zone')

    if workers:
        invalid_workers = set(workers).difference(set(actual_workers))
        assert len(invalid_workers) == 0, "Non-existent workers specified: " + ", ".join(invalid_workers)

    if take:
        assert 0 < take <= len(workers), \
            "Number of workers to take must be in the range of [0, nWorkers]. Found " + take + "."
        workers = workers[:take]

    def gcloud_ssh(remote, command):
        return 'gcloud compute ssh {remote} --zone {zone} --command "{command}"'.format(remote=remote, zone=zone,
                                                                                        command=command)

    def gcloud_copy_files(remote, src, dest):
        return 'gcloud compute copy-files {remote}:{src} {dest} --zone {zone}'.format(remote=remote, src=src, dest=dest,
                                                                                      zone=zone)

    def gsutil_cp(src, dest):
        return 'gsutil -m cp -r {src} {dest}'.format(src=src, dest=dest)

    def copy_files_tmp(remote, files, dest, tmp):
        init_cmd = ['mkdir -p {tmp}; rm -r {tmp}/*'.format(tmp=tmp)]

        copy_tmp_cmds = ['sudo cp -r {file} {tmp}'.format(file=file, tmp=tmp) for file in files]
        copy_tmp_cmds.append('sudo chmod -R 777 {tmp}'.format(tmp=tmp))

        if compress:
            copy_tmp_cmds.append('sudo find ' + tmp + ' -type f ! -name \'*.gz\' -exec gzip "{}" \\;')

        call(gcloud_ssh(remote, '; '.join(init_cmd + copy_tmp_cmds)), shell=True)

        if not is_local:
            copy_dest_cmd = gcloud_ssh(remote, 'gsutil -m cp -r {tmp} {dest}'.format(tmp=tmp, dest=dest))
        else:
            copy_dest_cmd = gcloud_copy_files(remote, tmp, dest)

        call(copy_dest_cmd, shell=True)

    if not no_diagnose:
        diagnose_tar_path = re.search(r'Diagnostic results saved in: (?P<tarfile>gs://\S+diagnostic\.tar)',
                                      str(Popen('gcloud dataproc clusters diagnose {name}'.format(name=cluster_name),
                                                shell=True,
                                                stdout=PIPE,
                                                stderr=PIPE).communicate())).group('tarfile')

        call(gsutil_cp(diagnose_tar_path, dest), shell=True)

    master_log_files = ['/var/log/hive/hive-*',
                        '/var/log/google-dataproc-agent.0.log',
                        '/var/log/dataproc-initialization-script-0.log',
                        '/var/log/hadoop-mapreduce/mapred-mapred-historyserver*',
                        '/var/log/hadoop-hdfs/*-m.*',
                        '/var/log/hadoop-yarn/yarn-yarn-resourcemanager-*-m.*',
                        hail_log
                        ]

    copy_files_tmp(master, master_log_files, master_dest, '/tmp/' + master + '/')

    worker_log_files = ['/var/log/hadoop-hdfs/hadoop-hdfs-datanode-*.*',
                        '/var/log/dataproc-startup-script.log',
                        '/var/log/hadoop-yarn/yarn-yarn-nodemanager-*.*']

    for worker in workers:
        copy_files_tmp(worker, worker_log_files, worker_dest, '/tmp/' + worker + '/')
        copy_files_tmp(worker, ['/var/log/hadoop-yarn/userlogs/'], dest, '/tmp/hadoop-yarn/')
