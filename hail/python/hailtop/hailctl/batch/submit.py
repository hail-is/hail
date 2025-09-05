import os
import shlex
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import orjson

import hailtop.batch as hb
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.config import (
    get_deploy_config,
    get_hail_config_path,
)

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
    volume_mounts: Optional[List[Tuple[str, str]]],
    shell: Optional[str],
    output: StructuredFormatPlusTextOption,
    wait: bool,
    quiet: bool,
    # FIXME: requester pays config
):
    async with AsyncExitStack() as exitstack:
        fs = RouterAsyncFS()
        exitstack.push_async_callback(fs.close)

        backend = hb.ServiceBackend(
            billing_project=billing_project,
            remote_tmpdir=remote_tmpdir,
            regions=regions,
            gcs_requester_pays_configuration=requester_pays_project,
        )

        exitstack.push_async_callback(backend.async_close)

        local_user_config_dir = '/hail-config/'

        b = hb.Batch(backend=backend, name=name, attributes=attributes)

        config_file_paths = await get_user_config_files(fs)
        config_file_inputs = [(os.path.basename(path), b.read_input(path)) for path in config_file_paths]
        config_file_str = "\n".join(
            f'mv {input} {local_user_config_dir}{file_name}' for file_name, input in config_file_inputs
        )

        volume_mount_inputs = []
        mkdirs_needed = {local_user_config_dir}

        for src, maybe_dest in volume_mounts:
            if await fs.isfile(src):
                if maybe_dest.endswith('/'):
                    local_dest = os.path.join(maybe_dest, os.path.basename(src))
                else:
                    local_dest = maybe_dest

                mkdirs_needed.add(os.path.dirname(local_dest))
                volume_mount_inputs.append((local_dest, b.read_input(src)))
            else:
                if not await fs.isdir(src):
                    raise ValueError(f'src "{src}" is not a directory.')
                if src.endswith('/') and not maybe_dest.endswith('/'):
                    raise ValueError('copy and renaming a directory is not supported.')

                if not src.endswith('/') and maybe_dest.endswith('/'):
                    dest = os.path.join(maybe_dest, os.path.basename(src))
                else:
                    dest = maybe_dest

                dest = dest.rstrip('/') + '/'

                volume_mount_paths = await get_volume_mount_files(fs, src)

                for path in volume_mount_paths:
                    path_relname = os.path.relpath(path, src)
                    if path_relname == '.':
                        path_relname = ''

                    local_dest = os.path.join(dest, path_relname)

                    mkdirs_needed.add(os.path.dirname(local_dest))
                    volume_mount_inputs.append((local_dest, b.read_input(path)))

        volume_mount_str = "\n".join(f'mv {input} {dest}' for dest, input in volume_mount_inputs)

        mkdirs_str = "\n".join(f'mkdir -p {dir}' for dir in mkdirs_needed)

        entrypoint_str = ' '.join([shq(x) for x in entrypoint])

        j = b.new_job(name=name or 'submit', attributes=attributes, shell=shell)

        j.image(image)

        _env = {
            'HAIL_QUERY_BACKEND': 'batch',
            'XDG_CONFIG_HOME': local_user_config_dir,
        }
        _env.update(env or {})

        for env_name, env_val in _env.items():
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


async def get_user_config_files(fs: RouterAsyncFS) -> List[str]:
    user_config_path = get_hail_config_path()
    if not user_config_path.exists():
        return []
    files = [await f.url() async for f in await fs.listfiles(str(user_config_path), recursive=False)]
    return [f for f in files if f.endswith('.ini')]


async def get_volume_mount_files(fs: RouterAsyncFS, src: str) -> List[str]:
    src = __real_absolute_local_path(src, strict=False)  # defer strictness checks and globbing
    if await fs.isdir(str(src)):
        return [await f.url() async for f in await fs.listfiles(str(src), recursive=True)]
    if await fs.isfile(str(src)):
        return [str(src)]
    raise ValueError(f'source `{src}` does not exist')


# Note well, friends:
# This uses the local environment to support paths with variables like $HOME or $XDG_ directories.
# Consequently, it is inappropriate to resolve paths on the worker with this function.
def __real_absolute_local_path(path: Union[str, os.PathLike[str]], *, strict: bool) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve(strict=strict)


def shq(p: Any) -> str:
    return shlex.quote(str(p))
