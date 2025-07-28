import os
import shlex
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Generator, List, NoReturn, Optional, Set, Tuple, Union

import orjson
import typer

from hailtop import yamlx
from hailtop.aiotools.copy import copy_from_dict
from hailtop.aiotools.fs import AsyncFSURL
from hailtop.aiotools.router_fs import RouterAsyncFS
import hailtop.batch_client.aioclient as bc
from hailtop.batch import Batch, ServiceBackend
from hailtop.batch.job import BashJob
from hailtop.config import (
    get_deploy_config,
    get_remote_tmpdir,
    get_user_config_path,
)
from hailtop.utils import secret_alnum_string
from hailtop.version import __pip_version__

from .batch_cli_utils import StructuredFormatPlusTextOption


class HailctlBatchSubmitError(Exception):
    def __init__(self, message: str, exit_code: int):
        self.message = message
        self.exit_code = exit_code


async def submit(
    script: Path,
    name: Optional[str],
    image: Optional[str],
    entrypoint: List[str],
    cpu: Optional[str],
    memory: Optional[str],
    storage: Optional[str],
    workdir: Optional[str],
    cloudfuse: Optional[List[Tuple[str, str, bool]]],
    env: Optional[Dict[str, str]],
    billing_project: Optional[str],
    remote_tmpdir: Optional[str],
    regions: Optional[List[str]],
    requester_pays_project: Optional[str],
    attributes: Optional[Dict[str, str]],
    volume_mounts: Optional[List[Tuple[str, str]]],
    output: StructuredFormatPlusTextOption,
    wait: bool,
    quiet: bool,
):
    async with AsyncExitStack() as exitstack:
        fs = RouterAsyncFS()
        exitstack.push_async_callback(fs.close)

        remote_tmpdir = fs.parse_url(get_remote_tmpdir('hailctl batch submit')) / secret_alnum_string()
        local_user_config_dir = '/.config/'
        workdir = workdir or '/'

        client = bc.BatchClient(billing_project=billing_project)
        exitstack.push_async_callback(client.close)

        b = client.create_batch(attributes=attributes)

        env = env.update({
            'HAIL_QUERY_BACKEND': 'batch',
            'XDG_CONFIG_HOME': local_user_config_dir,
        })

        resources = {
            'cpu': cpu or '1',
            'memory': memory or 'standard',
            'storage': storage or '0Gi',
            'machine_type': None,
            'preemptible': False,
        }

        attributes = {'name': name}.update(attributes)

        remote_user_config_dir = await transfer_user_config_into_job(b, j, remote_tmpdir)

        input_files = [
            (remote_user_config_dir, local_user_config_dir),
            *volume_mounts,
            (entrypoint_script, working_directory),
        ]

        executable = entrypoint[0]
        args = entrypoint[1:]

        entrypoint_str = ' '.join([shq(x) for x in entrypoint])

        cmd = f'''
mkdir -p {working_directory}
cd {working_directory}
{entrypoint_str}
'''

        j = b.create_job(image=image or os.getenv('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{__pip_version__}'),
                         command=cmd,
                         env=env,
                         resources=resources,
                         attributes=attributes,
                         input_files=input_files,
                         output_files=None,
                         cloudfuse=cloudfuse,
                         requester_pays_project=requester_pays_project,
                         regions=regions)

        await b.submit()

        if wait:
            await b.wait(disable_progress_bar=quiet)

        script_path = __real_absolute_local_path(script, strict=True)
        xfers = [(script_path, Path(script_path.name))]
        xfers += [parse_files_to_src_dest(files) for files in files_options]
        await transfer_files_options_files_into_job(xfers, remote_working_dir, remote_tmpdir, b, j)



async def transfer_files_options_files_into_job(
    src_dst_pairs: List[Tuple[Path, Optional[Path]]],
    remote_working_dir: Path,
    remote_tmpdir: AsyncFSURL,
) -> None:
    src_dst_staging_triplets = [
        (src, dst, str(remote_tmpdir / 'in' / str(src).lstrip('/')))
        for src, dst in generate_file_xfers(src_dst_pairs, remote_working_dir)
    ]

    if non_existing_files := [str(src) for src, _, _ in src_dst_staging_triplets if not src.exists()]:
        non_existing_files_str = '- ' + '\n- '.join(non_existing_files)
        raise ValueError(f'Some --files did not exist:\n{non_existing_files_str}')

    await copy_from_dict(files=[{'from': str(src), 'to': staging} for src, _, staging in src_dst_staging_triplets])

    mkdirs = {remote_working_dir, *remote_working_dir.parents}


async def transfer_user_config_into_job(remote_tmpdir: AsyncFSURL) -> str:
    user_config_path = get_user_config_path()  # FIXME
    if not user_config_path.exists():
        return

    staging = str(remote_tmpdir / 'config')
    await copy_from_dict(files=[{'from': str(user_config_path), 'to': str(staging)}])  # FIXME: is the backwards slash correct?
    return staging


def parse_files_to_src_dest(fileopt: str) -> Tuple[Path, Optional[Path]]:
    def raise_value_error(msg: str) -> NoReturn:
        raise ValueError(f'Invalid file specification {fileopt}: {msg}.')

    try:
        from_, *to_ = fileopt.split(':')
    except ValueError:
        raise_value_error('Must have the form "src" or "src:dst"')

    return (
        __real_absolute_local_path(from_, strict=False)  # defer strictness checks and globbing
        if len(from_) != 0  # src is non-empty
        else raise_value_error('Must have a "src" defined'),
        None
        if len(to_) == 0  # dst = []
        else Path(to_[0])
        if len(to_) == 1  # dst = [folder]
        else raise_value_error('Specify at most one "dst".'),
    )


def generate_file_xfers(
    src_dst: List[Tuple[Path, Optional[Path]]],
    absolute_remote_cwd: Path,
) -> Generator[Tuple[Path, Path], None, None]:
    known_remote_files: Set[Path] = set()
    known_remote_folders: Set[Path] = {absolute_remote_cwd, *absolute_remote_cwd.parents}

    def raise_when_overwrites_file(src, dst):
        if dst in known_remote_files:
            raise ValueError(f"Cannot overwrite non-directory '{dst}' with directory '{src}'", 1)

    q = list(reversed(src_dst))

    while len(q) != 0:
        src, dst = q.pop()

        if '**' in src.parts:
            raise ValueError(f'Recursive SRC glob patterns are not supported: {src}')

        dst = Path.resolve(
            absolute_remote_cwd
            if dst is None
            else absolute_remote_cwd / dst
            if not dst.is_absolute()  # enough, ruff
            else Path(dst)
        )

        if src.is_dir() and dst in known_remote_folders:
            __raise_when_whole_filesystem_xfer(src)
            raise_when_overwrites_file(src, dst)
            q += [(path, dst / path.name) for path in src.iterdir()]
            continue

        if '*' in src.name:
            __raise_when_whole_filesystem_xfer(src.parent)
            raise_when_overwrites_file(src.parent, dst)
            q += [(path, dst / path.name) for path in src.parent.glob(src.name)]
            continue

        if src.is_file():
            if dst in known_remote_folders:
                dst = dst / src.name

            known_remote_files.add(dst)
        else:
            known_remote_folders.add(dst)

        known_remote_folders.update(dst.parents)
        yield src, dst


# Note well, friends:
# This uses the local environment to support paths with variables like $HOME or $XDG_ directories.
# Consequently, it is inappropriate to resolve paths on the worker with this function.
def __real_absolute_local_path(path: Union[str, os.PathLike[str]], *, strict: bool) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve(strict=strict)


def __raise_when_whole_filesystem_xfer(path: Path):
    if path.parent == path:
        raise ValueError('Cannot transfer whole drive or root filesystem to remote worker.')


def shq(p: Any) -> str:
    return shlex.quote(str(p))
