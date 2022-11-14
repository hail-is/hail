import re
import json
from subprocess import call, Popen, PIPE


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--dest', '-d', required=True, type=str, help="Directory for diagnose output -- must be local.")
    parser.add_argument('--hail-log', '-l', required=False, type=str, default='/home/hail/hail.log',
                        help="Path for hail.log file.")
    parser.add_argument('--overwrite', required=False, action='store_true',
                        help="Delete dest directory before adding new files.")
    parser.add_argument('--no-diagnose', required=False, action='store_true',
                        help="Do not run gcloud dataproc clusters diagnose.")
    parser.add_argument('--compress', '-z', required=False, action='store_true', help="GZIP all files.")
    parser.add_argument('--workers', required=False, nargs='*', help="Specific workers to get log files from.")
    parser.add_argument('--take', required=False, type=int, default=None,
                        help="Only download logs from the first N workers.")


async def main(args, pass_through_args):  # pylint: disable=unused-argument
    print("Diagnosing cluster '{}'...".format(args.name))

    is_local = not args.dest.startswith("gs://")

    if args.overwrite:
        if is_local:
            call('rm -r {dir}'.format(dir=args.dest), shell=True)
        else:
            call('gsutil -m rm -r {dir}'.format(dir=args.dest), shell=True)

    master_dest = args.dest.rstrip('/') + "/master/"
    worker_dest = args.dest.rstrip('/') + "/workers/"

    if is_local:
        call('mkdir -p {dir}'.format(dir=master_dest), shell=True)
        call('mkdir -p {dir}'.format(dir=worker_dest), shell=True)

    with Popen('gcloud dataproc clusters describe {name} --format json'.format(name=args.name),
               shell=True,
               stdout=PIPE,
               stderr=PIPE) as process:
        desc = json.loads(process.communicate()[0].strip())

    config = desc['config']

    master = config['masterConfig']['instanceNames'][0]
    try:
        workers = config['workerConfig']['instanceNames'] + config['secondaryWorkerConfig']['instanceNames']
    except KeyError:
        workers = config['workerConfig']['instanceNames']
    zone_match = re.search(r'zones/(?P<zone>\S+)$', config['gceClusterConfig']['zoneUri'])
    assert zone_match
    zone = zone_match.group('zone')

    if args.workers:
        invalid_workers = set(args.workers).difference(set(workers))
        assert len(invalid_workers) == 0, "Non-existent workers specified: " + ", ".join(invalid_workers)
        workers = args.workers

    if args.take:
        assert args.take > 0 and args.take <= len(
            workers), "Number of workers to take must be in the range of [0, nWorkers]. Found " + args.take + "."
        workers = workers[:args.take]

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

        if args.compress:
            copy_tmp_cmds.append('sudo find ' + tmp + ' -type f ! -name \'*.gz\' -exec gzip "{}" \\;')

        call(gcloud_ssh(remote, '; '.join(init_cmd + copy_tmp_cmds)), shell=True)

        if not is_local:
            copy_dest_cmd = gcloud_ssh(remote, 'gsutil -m cp -r {tmp} {dest}'.format(tmp=tmp, dest=dest))
        else:
            copy_dest_cmd = gcloud_copy_files(remote, tmp, dest)

        call(copy_dest_cmd, shell=True)

    if not args.no_diagnose:
        with Popen('gcloud dataproc clusters diagnose {name}'.format(name=args.name),
                   shell=True,
                   stdout=PIPE,
                   stderr=PIPE) as process:
            output = process.communicate()
        diagnose_tar_path_match = re.search(r'Diagnostic results saved in: (?P<tarfile>gs://\S+diagnostic\.tar)', str(output))
        assert diagnose_tar_path_match
        diagnose_tar_path = diagnose_tar_path_match.group('tarfile')

        call(gsutil_cp(diagnose_tar_path, args.dest), shell=True)

    master_log_files = ['/var/log/hive/hive-*',
                        '/var/log/google-dataproc-agent.0.log',
                        '/var/log/dataproc-initialization-script-0.log',
                        '/var/log/hadoop-mapreduce/mapred-mapred-historyserver*',
                        '/var/log/hadoop-hdfs/*-m.*',
                        '/var/log/hadoop-yarn/yarn-yarn-resourcemanager-*-m.*',
                        args.hail_log
                        ]

    copy_files_tmp(master, master_log_files, master_dest, '/tmp/' + master + '/')

    worker_log_files = ['/var/log/hadoop-hdfs/hadoop-hdfs-datanode-*.*',
                        '/var/log/dataproc-startup-script.log',
                        '/var/log/hadoop-yarn/yarn-yarn-nodemanager-*.*']

    for worker in workers:
        copy_files_tmp(worker, worker_log_files, worker_dest, '/tmp/' + worker + '/')
        copy_files_tmp(worker, ['/var/log/hadoop-yarn/userlogs/'], args.dest, '/tmp/hadoop-yarn/')
