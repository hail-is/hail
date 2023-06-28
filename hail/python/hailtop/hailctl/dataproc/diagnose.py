import re
import json

from typing import List, Optional
from subprocess import call, Popen, PIPE


def diagnose(
    name: str,
    dest: str,
    hail_log: str,
    overwrite: bool,
    no_diagnose: bool,
    compress: bool,
    workers: List[str],
    take: Optional[int],
):
    print("Diagnosing cluster '{}'...".format(name))

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

    with Popen(
        'gcloud dataproc clusters describe {name} --format json'.format(name=name), shell=True, stdout=PIPE, stderr=PIPE
    ) as process:
        desc = json.loads(process.communicate()[0].strip())

    config = desc['config']

    master = config['masterConfig']['instanceNames'][0]
    try:
        all_workers = config['workerConfig']['instanceNames'] + config['secondaryWorkerConfig']['instanceNames']
    except KeyError:
        all_workers = config['workerConfig']['instanceNames']
    zone_match = re.search(r'zones/(?P<zone>\S+)$', config['gceClusterConfig']['zoneUri'])
    assert zone_match
    zone = zone_match.group('zone')

    if workers:
        invalid_workers = set(workers).difference(set(all_workers))
        if invalid_workers:
            raise ValueError("Non-existent workers specified: " + ", ".join(invalid_workers))

    if take:
        if take < 0 or take > len(workers):
            raise ValueError(f'Number of workers to take must be in the range of [0, nWorkers]. Found {take}.')
        workers = workers[:take]

    def gcloud_ssh(remote, command):
        return 'gcloud compute ssh {remote} --zone {zone} --command "{command}"'.format(
            remote=remote, zone=zone, command=command
        )

    def gcloud_copy_files(remote, src, dest):
        return 'gcloud compute copy-files {remote}:{src} {dest} --zone {zone}'.format(
            remote=remote, src=src, dest=dest, zone=zone
        )

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
        with Popen(
            'gcloud dataproc clusters diagnose {name}'.format(name=name), shell=True, stdout=PIPE, stderr=PIPE
        ) as process:
            output = process.communicate()
        diagnose_tar_path_match = re.search(
            r'Diagnostic results saved in: (?P<tarfile>gs://\S+diagnostic\.tar)', str(output)
        )
        assert diagnose_tar_path_match
        diagnose_tar_path = diagnose_tar_path_match.group('tarfile')

        call(gsutil_cp(diagnose_tar_path, dest), shell=True)

    master_log_files = [
        '/var/log/hive/hive-*',
        '/var/log/google-dataproc-agent.0.log',
        '/var/log/dataproc-initialization-script-0.log',
        '/var/log/hadoop-mapreduce/mapred-mapred-historyserver*',
        '/var/log/hadoop-hdfs/*-m.*',
        '/var/log/hadoop-yarn/yarn-yarn-resourcemanager-*-m.*',
        hail_log,
    ]

    copy_files_tmp(master, master_log_files, master_dest, '/tmp/' + master + '/')

    worker_log_files = [
        '/var/log/hadoop-hdfs/hadoop-hdfs-datanode-*.*',
        '/var/log/dataproc-startup-script.log',
        '/var/log/hadoop-yarn/yarn-yarn-nodemanager-*.*',
    ]

    for worker in workers:
        copy_files_tmp(worker, worker_log_files, worker_dest, '/tmp/' + worker + '/')
        copy_files_tmp(worker, ['/var/log/hadoop-yarn/userlogs/'], dest, '/tmp/hadoop-yarn/')
