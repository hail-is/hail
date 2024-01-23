import orjson
import os
import re
from shlex import quote as shq
from hailtop import pip_version
from typing import Tuple, Optional, List
import typer
from contextlib import AsyncExitStack
import hailtop.batch as hb
from hailtop.batch.job import Job
from hailtop.aiotools.router_fs import RouterAsyncFS, AsyncFSURL
from hailtop.aiotools.copy import copy_from_dict
from hailtop.config import (
    get_remote_tmpdir,
    get_user_config_path,
    get_deploy_config,
)
from hailtop.utils import (
    secret_alnum_string,
    unpack_comma_delimited_inputs,
)

from .batch_cli_utils import StructuredFormatPlusTextOption


def real_absolute_expanded_path(path: str) -> Tuple[str, bool]:
    had_trailing_slash = path[-1] == '/'  # NB: realpath removes trailing slash
    return os.path.realpath(os.path.abspath(os.path.expanduser(path))), had_trailing_slash


def real_absolute_cwd() -> str:
    return real_absolute_expanded_path(os.getcwd())[0]


class HailctlBatchSubmitError(Exception):
    def __init__(self, message: str, exit_code: int):
        self.message = message
        self.exit_code = exit_code


async def submit(
    name: str,
    image_name: Optional[str],
    files_options: List[str],
    output: StructuredFormatPlusTextOption,
    script: str,
    arguments: List[str],
    wait: bool,
):
    files_options = unpack_comma_delimited_inputs(files_options)

    quiet = output != 'text'

    async with AsyncExitStack() as exitstack:
        fs = RouterAsyncFS()
        exitstack.push_async_callback(fs.close)

        remote_tmpdir = fs.parse_url(get_remote_tmpdir('hailctl batch submit')).with_new_path_component(
            secret_alnum_string()
        )

        backend = hb.ServiceBackend()
        exitstack.push_async_callback(backend._async_close)

        b = hb.Batch(name=name, backend=backend)
        j = b.new_bash_job()
        j.image(image_name or os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}'))
        j.env('HAIL_QUERY_BACKEND', 'batch')

        await transfer_files_options_files_into_job(remote_tmpdir, files_options, j, b)

        script_cloud_file, user_config_cloud_file = await upload_script_and_user_config(remote_tmpdir, script)
        if user_config_cloud_file is not None:
            config_file = b.read_input(user_config_cloud_file)
            j.command('mkdir -p $HOME/.config/hail')
            j.command(f'ln -s {shq(config_file)} $HOME/.config/hail/config.ini')

        real_cwd = real_absolute_cwd()
        j.command(f'mkdir -p {shq(real_cwd)}')
        j.command(f'cd {shq(real_cwd)}')

        command = 'python3' if script.endswith('.py') else 'bash'
        script_file = b.read_input(script_cloud_file)
        script_arguments = " ".join(shq(x) for x in arguments)
        j.command(f'{command} {script_file} {script_arguments}')

        batch_handle = await b._async_run(wait=False, disable_progress_bar=quiet)
        assert batch_handle

        if output == 'text':
            deploy_config = get_deploy_config()
            url = deploy_config.external_url('batch', f'/batches/{batch_handle.id}/jobs/1')
            print(f'Submitted batch {batch_handle.id}, see {url}')
        else:
            assert output == 'json'
            print(orjson.dumps({'id': batch_handle.id}).decode('utf-8'))

        if wait:
            out = batch_handle.wait(disable_progress_bar=quiet)
            if output == 'text':
                print(out)
            else:
                print(orjson.dumps(out))
            if out['state'] != 'success':
                raise typer.Exit(1)


def cloud_prefix(remote_tmpdir: AsyncFSURL, path: str) -> str:
    path = path.lstrip('/')
    return str(remote_tmpdir.with_new_path_component(path))


FILE_REGEX = re.compile(r'([^:]+)(?::(.+))?')


def parse_files_option_to_src_dest_and_cloud_intermediate(remote_tmpdir: AsyncFSURL, file: str) -> Tuple[str, str, str]:
    match = FILE_REGEX.match(file)
    if match is None:
        raise ValueError(f'invalid file specification {file}. Must have the form "src" or "src:dest"')

    src, dest = match.groups()

    if src is None:
        raise ValueError(f'invalid file specification {file}. Must have a "src" defined.')

    src, src_looks_like_directory = real_absolute_expanded_path(src)

    if dest is None:
        dest = os.path.join(real_absolute_cwd(), os.path.basename(src))
    else:
        dest, dest_looks_like_directory = real_absolute_expanded_path(dest)
        if not src_looks_like_directory and dest_looks_like_directory:
            dest = os.path.join(dest, os.path.basename(src))

    return (src, dest, cloud_prefix(remote_tmpdir, src))


async def transfer_files_options_files_into_job(
    remote_tmpdir: AsyncFSURL, files_options: List[str], j: Job, b: hb.Batch
):
    src_dst_cloud_intermediate_triplets = [
        parse_files_option_to_src_dest_and_cloud_intermediate(remote_tmpdir, files_option)
        for files_option in files_options
    ]

    if non_existing_files := [src for src, _, _ in src_dst_cloud_intermediate_triplets if not os.path.exists(src)]:
        non_existing_files_str = '- ' + '\n- '.join(non_existing_files)
        raise HailctlBatchSubmitError(f'Some --files did not exist:\n{non_existing_files_str}', 1)

    await copy_from_dict(
        files=[
            {'from': src, 'to': cloud_intermediate}
            for src, _, cloud_intermediate in src_dst_cloud_intermediate_triplets
        ]
    )

    for _, dest, cloud_intermediate in src_dst_cloud_intermediate_triplets:
        in_file = b.read_input(cloud_intermediate)
        j.command(f'mkdir -p {shq(os.path.dirname(dest))}; ln -s {shq(in_file)} {shq(dest)}')


async def upload_script_and_user_config(remote_tmpdir: AsyncFSURL, script: str):
    if not os.path.exists(script):
        raise HailctlBatchSubmitError(f'Script file does not exist: {script}', 1)
    script_src, _, script_cloud_file = parse_files_option_to_src_dest_and_cloud_intermediate(remote_tmpdir, script)

    extra_files_to_copy = [
        # In Azure, two concurrent uploads to the same blob path race until one succeeds. The
        # other fails with an error that is indistinguishable from a client-side logic
        # error. We could treat "Invalid Block List" as a limited retry error, but, currently,
        # multi-part-create does not retry when an error occurs in the `__aexit__` which is
        # when we would experience the error for multi-part-creates.
        # https://azure.github.io/Storage/docs/application-and-user-data/code-samples/concurrent-uploads-with-versioning/
        # https://github.com/hail-is/hail/pull/13812#issuecomment-1882088862
        {'from': script_src, 'to': script_cloud_file},
    ]

    user_config_path = str(get_user_config_path())
    user_config_cloud_file = None
    if os.path.exists(user_config_path):
        user_config_src, _, user_config_cloud_file = parse_files_option_to_src_dest_and_cloud_intermediate(
            remote_tmpdir, user_config_path
        )
        extra_files_to_copy.append({'from': user_config_src, 'to': user_config_cloud_file})

    await copy_from_dict(files=extra_files_to_copy)
    return script_cloud_file, user_config_cloud_file
