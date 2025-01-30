import os
import shlex
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple

import orjson

import hailtop.batch as hb
from hailtop.config import (
    get_deploy_config,
    get_hail_config_path,
)

from .batch_cli_utils import StructuredFormatPlusTextOption


class HailctlBatchSubmitError(Exception):
    def __init__(self, message: str, exit_code: int):
        self.message = message
        self.exit_code = exit_code


def submit(
    image: str,
    entrypoint: List[str],
    name: Optional[str],
    cpu: str,
    memory: str,
    storage: str,
    machine_type: Optional[str],
    spot: bool,
    workdir: str,
    cloudfuse: Optional[List[Tuple[str, str, bool]]],
    env: Optional[Dict[str, str]],
    billing_project: Optional[str],
    remote_tmpdir: Optional[str],
    regions: Optional[List[str]],
    requester_pays_project: Optional[str],
    attributes: Optional[Dict[str, str]],
    volume_mounts: Optional[List[Tuple[str, str]]],
    shell: Optional[str],
    output: StructuredFormatPlusTextOption,
    wait: bool,
    quiet: bool,
    # FIXME: requester pays config
):
    with hb.ServiceBackend(
        billing_project=billing_project,
        remote_tmpdir=remote_tmpdir,
        regions=regions,
        gcs_requester_pays_configuration=requester_pays_project,
    ) as backend:
        job_hail_config_dir = PurePath('/hail-config/hail')

        b = hb.Batch(backend=backend, name=name, attributes=attributes)

        config_file_inputs = [(path.name, b.read_input(path)) for path in get_user_config_files()]
        config_file_str = "\n".join(
            f'mv {batch_input} {job_hail_config_dir / file_name}' for file_name, batch_input in config_file_inputs
        )

        volume_mount_inputs = []
        mkdirs_needed = {job_hail_config_dir}

        for str_src, maybe_dest in volume_mounts or []:
            src = Path(str_src).expanduser()
            if src.is_file():
                if maybe_dest.endswith('/'):
                    local_dest = PurePath(maybe_dest, src.name)
                else:
                    local_dest = PurePath(maybe_dest)

                mkdirs_needed.add(local_dest.parent)
                volume_mount_inputs.append((local_dest, b.read_input(src)))
            else:
                if not src.is_dir():
                    raise ValueError(f'src "{src}" is not a directory.')
                if str_src.endswith('/') and not maybe_dest.endswith('/'):
                    raise ValueError('copy and renaming a directory is not supported.')

                if not str_src.endswith('/') and maybe_dest.endswith('/'):
                    dest = PurePath(maybe_dest, src.name)
                else:
                    dest = PurePath(maybe_dest)

                for curdir, _dirnames, filenames in os.walk(src, followlinks=True):
                    for file in filenames:
                        path = Path(curdir, file).relative_to(src)
                        local_dest = dest / path

                        mkdirs_needed.add(local_dest.parent)
                        volume_mount_inputs.append((local_dest, b.read_input(path)))

        volume_mount_str = "\n".join(f'mv {input} {dest}' for dest, input in volume_mount_inputs)

        mkdirs_str = "\n".join(f'mkdir -p {dir}' for dir in mkdirs_needed)

        entrypoint_str = ' '.join([shq(x) for x in entrypoint])

        j = b.new_job(name=name or 'submit', attributes=attributes, shell=shell)

        j.image(image)

        job_env = {
            'HAIL_QUERY_BACKEND': 'batch',
            'XDG_CONFIG_HOME': os.fspath(job_hail_config_dir.parent),
        }
        job_env.update(env or {})

        for env_name, env_val in job_env.items():
            j.env(env_name, env_val)

        j.cpu(cpu)
        j.memory(memory)
        j.storage(storage)
        j.spot(spot)
        j._machine_type = machine_type

        workdir = workdir or '/'

        j.command(f"""
{mkdirs_str}
{config_file_str}
{volume_mount_str}
cd {workdir}
{entrypoint_str}
""")

        if cloudfuse is not None:
            for bucket, mount, read_only in cloudfuse:
                j.cloudfuse(bucket, mount, read_only=read_only)

        bc_b = b.run(wait=False, disable_progress_bar=True)
        assert bc_b is not None  # needed for typechecking

        if output == 'text':
            deploy_config = get_deploy_config()
            url = deploy_config.external_url('batch', f'/batches/{bc_b.id}/jobs/1')
            print(f'Submitted batch {bc_b.id}, see {url}')
        # FIXME: support yaml
        else:
            assert output == 'json'
            print(orjson.dumps({'id': bc_b.id}).decode('utf-8'))

        if wait:
            bc_b.wait(disable_progress_bar=quiet)


def get_user_config_files() -> List[Path]:
    user_config_path = get_hail_config_path()
    if not user_config_path.exists():
        return []
    return [item for item in user_config_path.iterdir() if item.suffix == '.ini']


def shq(p: Any) -> str:
    return shlex.quote(str(p))
