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
    files_options: List[str],
    output: StructuredFormatPlusTextOption,
    wait: bool,
    quiet: bool,
    *arguments: str,
):
    async with AsyncExitStack() as exitstack:
        fs = RouterAsyncFS()
        exitstack.push_async_callback(fs.close)

        remote_tmpdir = fs.parse_url(get_remote_tmpdir('hailctl batch submit')) / secret_alnum_string()

        backend = ServiceBackend()
        exitstack.push_async_callback(backend._async_close)

        b = Batch(name=name, backend=backend)
        j = b.new_job(shell='/bin/sh')
        j.image(image or os.getenv('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{__pip_version__}'))
        j.env('HAIL_QUERY_BACKEND', 'batch')
        await transfer_user_config_into_job(b, j, remote_tmpdir)

        # The knowledge of why the current working directory is mirrored onto the
        # worker has been lost to the sands of time. Some speculate that a user's
        # code didn't work because it relied on the state of the local filesystem.
        # Nonetheless, we continue the fine work of our forebears like good sheep.
        remote_working_dir = __real_absolute_local_path('.', strict=True)
        j.command(f'mkdir -p {shq(remote_working_dir)} && cd {shq(remote_working_dir)}')

        script_path = __real_absolute_local_path(script, strict=True)
        xfers = [(script_path, Path(script_path.name))]
        xfers += [parse_files_to_src_dest(files) for files in files_options]
        await transfer_files_options_files_into_job(xfers, remote_working_dir, remote_tmpdir, b, j)

        executable = shq(script_path.name)
        j.name = executable

        command = 'python3' if executable.endswith(".py") else f'chmod +x ./{executable} &&'
        shargs = ' '.join([shq(x) for x in arguments])
        j.command(f'{command} ./{executable} {shargs}')

        # Mix `async` calls with those that internally use `async_to_blocking` at your own peril.
        batch_handle = await b._async_run(wait=False, disable_progress_bar=quiet)
        assert batch_handle
        async_batch = batch_handle._async_batch

        if not wait:
            deploy_config = get_deploy_config()
            url = deploy_config.external_url('batch', f'/batches/{async_batch.id}/jobs/1')
            print(
                f'Submitted batch {async_batch.id}, see {url}.'
                if output == 'text'
                else orjson.dumps({'batch_id': async_batch.id, 'url': url}).decode('utf-8')
                if output == 'json'
                else yamlx.dump({'batch_id': async_batch.id, 'url': url})
            )
        else:
            out = await async_batch.wait(disable_progress_bar=quiet)
            try:
                out['log'] = (await async_batch.get_job_log(1))['main']
            except Exception as e:
                out['log'] = f'Could not retrieve job log: {e}'

            print(orjson.dumps(out).decode('utf-8') if output == 'json' else yamlx.dump(out))

            if out['state'] != 'success':
                raise typer.Exit(1)


async def transfer_files_options_files_into_job(
    src_dst_pairs: List[Tuple[Path, Optional[Path]]],
    remote_working_dir: Path,
    remote_tmpdir: AsyncFSURL,
    b: Batch,
    j: BashJob,
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

    for _, dst, staging in src_dst_staging_triplets:
        in_file = await b._async_read_input(staging)

        if dst.parent not in mkdirs:
            j.command(f'mkdir -p {shq(dst.parent)}')
            mkdirs.update(dst.parents)

        j.command(f'ln -sT {shq(in_file)} {shq(dst)}')


async def transfer_user_config_into_job(b: Batch, j: BashJob, remote_tmpdir: AsyncFSURL) -> None:
    user_config_path = get_user_config_path()
    if not user_config_path.exists():
        return

    staging = str(remote_tmpdir / user_config_path.name)
    await copy_from_dict(files=[{'from': str(user_config_path), 'to': str(staging)}])
    file = await b._async_read_input(staging)
    j.command(f'mkdir -p $HOME/.config/hail && ln -sT {file} $HOME/.config/hail/config.ini')


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
