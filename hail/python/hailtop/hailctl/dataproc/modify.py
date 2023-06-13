import os.path
import sys

from typing import List, Optional

from . import gcloud
from .deploy_metadata import get_deploy_metadata


def modify(
    name: str,
    num_workers: Optional[int],
    num_secondary_workers: Optional[int],
    graceful_decommission_timeout: Optional[str],
    max_idle: Optional[str],
    no_max_idle: bool,
    expiration_time: Optional[str],
    max_age: Optional[str],
    no_max_age: bool,
    dry_run: bool,
    zone: Optional[str],
    update_hail_version: bool,
    wheel: Optional[str],
    beta: bool,
    pass_through_args: List[str],
):
    modify_args = []
    if num_workers is not None:
        modify_args.append('--num-workers={}'.format(num_workers))

    if num_secondary_workers is not None:
        modify_args.append('--num-secondary-workers={}'.format(num_secondary_workers))

    if graceful_decommission_timeout:
        if not modify_args:
            sys.exit("Error: Cannot use --graceful-decommission-timeout without resizing the cluster.")
        modify_args.append('--graceful-decommission-timeout={}'.format(graceful_decommission_timeout))

    if max_idle:
        modify_args.append('--max-idle={}'.format(max_idle))
    if no_max_idle:
        modify_args.append('--no-max-idle')
    if expiration_time:
        modify_args.append('--expiration_time={}'.format(expiration_time))
    if max_age:
        modify_args.append('--max-age={}'.format(max_age))
    if no_max_age:
        modify_args.append('--no-max-age')

    if modify_args:
        cmd = ['dataproc', 'clusters', 'update', name, *modify_args]

        if beta:
            cmd.insert(0, 'beta')

        cmd.extend(pass_through_args)

        # print underlying gcloud command
        print('gcloud ' + ' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join(cmd[4:]))

        # Update cluster
        if not dry_run:
            print("Updating cluster '{}'...".format(name))
            gcloud.run(cmd)

    if update_hail_version and wheel is not None:
        sys.exit('argument --update-hail-version: not allowed with argument --wheel')

    if update_hail_version:
        deploy_metadata = get_deploy_metadata()
        wheel = deploy_metadata["wheel"]

    if wheel is not None:
        zone = zone if zone else gcloud.get_config("compute/zone")
        if not zone:
            raise RuntimeError(
                "Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default."
            )

        wheelfile = os.path.basename(wheel)
        cmds = []
        if wheel.startswith("gs://"):
            cmds.append(
                [
                    'compute',
                    'ssh',
                    '{}-m'.format(name),
                    '--zone={}'.format(zone),
                    '--',
                    f'sudo gsutil cp {wheel} /tmp/ && '
                    'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                    f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile} && '
                    f"unzip /tmp/{wheelfile} && "
                    "requirements_file=$(mktemp) && "
                    "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' >$requirements_file &&"
                    "/opt/conda/default/bin/pip install -r $requirements_file",
                ]
            )
        else:
            cmds.extend(
                [
                    ['compute', 'scp', '--zone={}'.format(zone), wheel, '{}-m:/tmp/'.format(name)],
                    [
                        'compute',
                        'ssh',
                        f'{name}-m',
                        f'--zone={zone}',
                        '--',
                        'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                        f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile} && '
                        f"unzip /tmp/{wheelfile} && "
                        "requirements_file=$(mktemp) && "
                        "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' >$requirements_file &&"
                        "/opt/conda/default/bin/pip install -r $requirements_file",
                    ],
                ]
            )

        for cmd in cmds:
            print('gcloud ' + ' '.join(cmd))
            if not dry_run:
                gcloud.run(cmd)

    if not wheel and not modify_args and pass_through_args:
        sys.stderr.write('ERROR: found pass-through arguments but not known modification args.')
        sys.exit(1)
