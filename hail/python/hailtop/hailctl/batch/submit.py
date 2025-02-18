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
    name: str,
    image: Optional[str],
    files_options: List[str],
    output: StructuredFormatPlusTextOption,
    script: str,
    arguments: List[str],
    wait: bool,
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
        working_dir_shq = shq(str(remote_working_dir))
        j.command(f'mkdir -p {working_dir_shq} && cd {working_dir_shq}')

        script_path = __real_absolute_local_path(script, strict=True)
        j.name = script_path.name

        xfers = [(script_path, script_path)]
        xfers += [parse_files_to_src_dest(files) for files in files_options]
        await transfer_files_options_files_into_job(xfers, remote_working_dir, remote_tmpdir, b, j)

        command = 'python3' if str(script_path).endswith('.py') else f'chmod +x {script_path} &&'
        script_arguments = " ".join(shq(x) for x in arguments)
        j.command(f'{command} {script_path} {script_arguments}')

        quiet = output != 'text'
        batch_handle = await b._async_run(wait=False, disable_progress_bar=quiet)
        assert batch_handle

        # You run into all sorts of problems mixing `async` calls with those
        # that internally use `async_to_blocking`. Until we split async/blocking
        # clients, use the "private" `_async_batch` property.
        async_batch = batch_handle._async_batch

        if output == 'text':
            deploy_config = get_deploy_config()
            url = deploy_config.external_url('batch', f'/batches/{async_batch.id}/jobs/1')
            print(f'Submitted batch {async_batch.id}, see {url}')
        else:
            assert output == 'json'
            print(orjson.dumps({'id': async_batch.id}).decode('utf-8'))

        if wait:
            out = await async_batch.wait(disable_progress_bar=quiet)
            try:
                out['log'] = (await async_batch.get_job_log(1))['main']
            except:
                out['log'] = 'Could not retrieve job log.'

            print(yamlx.dump(out) if output == 'text' else orjson.dumps(out))

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
        raise HailctlBatchSubmitError(f'Some --files did not exist:\n{non_existing_files_str}', 1)

    await copy_from_dict(files=[{'from': str(src), 'to': staging} for src, _, staging in src_dst_staging_triplets])

    parents = list({dst.parent for _, dst, _ in src_dst_staging_triplets})
    parents.sort(key=lambda p: len(p.parts), reverse=True)
    mkdirs = {remote_working_dir, *remote_working_dir.parents}
    for folder in parents:
        if folder not in mkdirs:
            j.command(f'mkdir -p {shq(folder)}')
            mkdirs.add(folder)
            mkdirs.update(folder.parents)

    for _, dst, staging in src_dst_staging_triplets:
        in_file = await b._async_read_input(staging)
        j.command(f'ln -s {shq(in_file)} {shq(dst)}')


async def transfer_user_config_into_job(b: Batch, j: BashJob, remote_tmpdir: AsyncFSURL) -> None:
    user_config_path = get_user_config_path()
    if not user_config_path.exists():
        return

    staging = str(remote_tmpdir / user_config_path.name)
    await copy_from_dict(files=[{'from': str(user_config_path), 'to': str(staging)}])
    file = await b._async_read_input(staging)
    j.command(f'mkdir -p $HOME/.config/hail && ln -s {file} $HOME/.config/hail/config.ini')


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
        else raise_value_error('specify at most one "dst".'),
    )


def generate_file_xfers(
    src_dst: List[Tuple[Path, Optional[Path]]],
    absolute_remote_cwd: Path,
) -> Generator[Tuple[Path, Path], None, None]:
    # Try to generate a set of deterministic SRC -> DST copy instructions
    # given a set of source (SRC) and (optional) destination paths (DST).

    visited: Set[Path] = set()
    known_paths: Set[Path] = set()
    q = list(reversed(src_dst))

    while len(q) != 0:
        src, dst = q.pop()

        if '**' in src.parts:
            raise HailctlBatchSubmitError(f'Recursive glob patterns are not supported: {src}', 1)

        dst = absolute_remote_cwd if dst is None else absolute_remote_cwd / dst if not dst.is_absolute() else dst

        if src.is_dir() or '*' in src.name:
            anchor, children = (src, src.iterdir()) if src.is_dir() else (src.parent, src.parent.glob(src.name))
            q += [(path, dst / path.relative_to(anchor)) for path in children]
            continue

        # assume src is a file
        if src in visited:
            continue

        if dst in known_paths:
            dst = dst / src.name

        visited.add(src)
        known_paths.add(dst)
        known_paths.update(dst.parents)

        yield src, dst


# Note well, friends:
# This uses the local environment to support paths with variables like $HOME or $XDG_ directories.
# Consequently, it is inappropriate to resolve paths on the worker with this function.
def __real_absolute_local_path(path: Union[str, Path], *, strict: bool) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve(strict=strict)


def shq(p: Any) -> str:
    return shlex.quote(str(p))
