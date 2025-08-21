import os
import shlex
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

import hailtop.batch_client.aioclient as bc
from hailtop.aiotools.copy import copy_from_dict
from hailtop.aiotools.fs import AsyncFSURL
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.config import (
    ConfigVariable,
    configuration_of,
    get_hail_config_path,
    get_remote_tmpdir,
)
from hailtop.utils import secret_alnum_string

from .batch_cli_utils import StructuredFormatPlusTextOption


class HailctlBatchSubmitError(Exception):
    def __init__(self, message: str, exit_code: int):
        self.message = message
        self.exit_code = exit_code


async def submit(
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
    volume_mounts: Optional[List[str]],
    shell: Optional[str],
    output: StructuredFormatPlusTextOption,
    wait: bool,
    quiet: bool,
):
    async with (AsyncExitStack() as exitstack):
        fs = RouterAsyncFS()
        exitstack.push_async_callback(fs.close)

        remote_tmpdir = fs.parse_url(get_remote_tmpdir('hailctl batch submit')) / secret_alnum_string()
        local_user_config_dir = '/io/hail-config/'
        workdir = workdir or '/'

        billing_project = configuration_of(ConfigVariable.BATCH_BILLING_PROJECT, billing_project, None)
        if billing_project is None:
            raise ValueError(
                'the billing_project parameter of ServiceBackend must be set '
                'or run `hailctl config set batch/billing_project '
                'MY_BILLING_PROJECT`'
            )

        client = await bc.BatchClient.create(billing_project=billing_project)
        exitstack.push_async_callback(client.close)

        b = client.create_batch(attributes=attributes)

        _env = {
            'HAIL_QUERY_BACKEND': 'batch',
            'XDG_CONFIG_HOME': local_user_config_dir,
        }
        _env.update(env or {})

        resources = {
            'cpu': cpu or '1',
            'memory': memory or 'standard',
            'storage': storage or '0Gi',
            'preemptible': spot,
        }

        if machine_type is not None:
            resources['machine_type'] = machine_type

        name = name or 'submit'
        _attributes: Dict[str, str] = {'name': name}
        _attributes.update(attributes or {})

        remote_user_config_dir = await transfer_user_config_into_job(remote_tmpdir)

        maybe_volume_mounts = await convert_volume_mounts_to_file_transfers_with_local_upload(fs, remote_tmpdir, volume_mounts)

        symlinks_needed = []
        input_files = [(remote_user_config_dir, local_user_config_dir + 'hail/')]

        for src, dest in maybe_volume_mounts:
            io_dest = f'/io{dest}'
            if not dest.startswith('/io/'):
                symlinks_needed.append((io_dest, dest))
            input_files.append((src, io_dest))

        entrypoint_str = ' '.join([shq(x) for x in entrypoint])

        symlinks = [f'ln -s {io_dest.rstrip("/")} {dest.rstrip("/")}' for io_dest, dest in symlinks_needed]
        symlinks_str = "\n".join(symlinks)

        cmd = f'''
{symlinks_str}
mkdir -p {workdir}
hailctl config list
hailctl config profile list
cd {workdir}
{entrypoint_str}
'''

        b.create_job(image=image,
                     command=[shell if shell else '/bin/bash', '-c', cmd],
                     env=_env,
                     resources=resources,
                     attributes=_attributes,
                     input_files=input_files,
                     output_files=None,
                     cloudfuse=cloudfuse,
                     requester_pays_project=requester_pays_project,
                     regions=regions)

        await b.submit()

        if wait:
            await b.wait(disable_progress_bar=quiet)


async def transfer_user_config_into_job(remote_tmpdir: AsyncFSURL) -> str:
    user_config_path = get_hail_config_path()
    if not user_config_path.exists():
        return

    staging = str(remote_tmpdir / 'hail-config')
    await copy_from_dict(files=[{'from': str(user_config_path), 'to': str(staging)}])
    return staging


def parse_to_src_dest(fileopt: str) -> Tuple[Path, Path]:
    def raise_value_error(msg: str) -> NoReturn:
        raise ValueError(f'Invalid file specification {fileopt}: {msg}.')

    try:
        from_, to_ = fileopt.split(':')
    except ValueError:
        raise_value_error('Must have the form "src:dst"')

    from_ = __real_absolute_local_path(from_, strict=False)  # defer strictness checks and globbing

    return (from_, to_)


async def convert_volume_mounts_to_file_transfers_with_local_upload(
    fs: RouterAsyncFS,
    remote_tmpdir: Path,
    volume_mounts: List[str],
) -> List[Tuple[str, str]]:
    local_file_transfers = []
    remote_file_transfers = []

    src_dst_list = [parse_to_src_dest(mount) for mount in volume_mounts]

    for src, dst in src_dst_list:
        if fs.parse_url(str(src)).scheme == 'file':
            remote_src = remote_tmpdir / 'input' / secret_alnum_string()
            local_file_transfers.append((src, remote_src))
        else:
            remote_src = src

        remote_file_transfers.append((str(remote_src), str(dst)))

    await copy_from_dict(files=[{'from': str(local_src), 'to': str(remote_src)}
                                for local_src, remote_src in local_file_transfers])

    return remote_file_transfers


# Note well, friends:
# This uses the local environment to support paths with variables like $HOME or $XDG_ directories.
# Consequently, it is inappropriate to resolve paths on the worker with this function.
def __real_absolute_local_path(path: Union[str, os.PathLike[str]], *, strict: bool) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve(strict=strict)


def shq(p: Any) -> str:
    return shlex.quote(str(p))
