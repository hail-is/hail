import orjson
import os
import re
from shlex import quote as shq
from hailtop import pip_version
from typing import Tuple, Optional, List
import typer
from contextlib import AsyncExitStack

from .batch_cli_utils import StructuredFormatPlusTextOption

FILE_REGEX = re.compile(r'(?P<src>[^:]+)(:(?P<dest>.+))?')


def real_absolute_expanded_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def real_absolute_cwd() -> str:
    return real_absolute_expanded_path(os.getcwd())


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
    import hailtop.batch as hb  # pylint: disable=import-outside-toplevel
    from hailtop.aiotools.copy import copy_from_dict  # pylint: disable=import-outside-toplevel
    from hailtop.config import (  # pylint: disable=import-outside-toplevel
        get_remote_tmpdir,
        get_user_config_path,
        get_deploy_config,
    )
    from hailtop.utils import (  # pylint: disable=import-outside-toplevel
        secret_alnum_string,
        unpack_comma_delimited_inputs,
    )

    files_options = unpack_comma_delimited_inputs(files_options)
    user_config_path = str(get_user_config_path())

    quiet = output != 'text'

    remote_tmpdir = get_remote_tmpdir('hailctl batch submit')
    remote_tmpdir = remote_tmpdir.rstrip('/')

    tmpdir_path_prefix = secret_alnum_string()

    def cloud_prefix(path):
        path = path.lstrip('/')
        return f'{remote_tmpdir}/{tmpdir_path_prefix}/{path}'

    def parse_files_option_to_src_dest_and_cloud_intermediate(file: str) -> Tuple[str, str, str]:
        match = FILE_REGEX.match(file)
        if match is None:
            raise ValueError(f'invalid file specification {file}. Must have the form "src" or "src:dest"')

        result = match.groupdict()

        src = result.get('src')
        if src is None:
            raise ValueError(f'invalid file specification {file}. Must have a "src" defined.')
        src = real_absolute_expanded_path(src)
        src = src.rstrip('/')

        dest = result.get('dest')
        if dest is not None:
            dest_intended_as_directory = dest[-1] == '/'
            dest = real_absolute_expanded_path(dest)
            if dest_intended_as_directory:
                dest = os.path.join(dest, os.path.basename(src))
        else:
            dest = os.path.join(real_absolute_cwd(), os.path.basename(src))

        cloud_file = cloud_prefix(src)

        return (src, dest, cloud_file)

    async with AsyncExitStack() as exitstack:
        backend = hb.ServiceBackend()
        exitstack.push_async_callback(backend._async_close)

        b = hb.Batch(name=name, backend=backend)
        j = b.new_bash_job()
        j.image(image_name or os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}'))

        src_dst_cloud_intermediate_triplets = [
            parse_files_option_to_src_dest_and_cloud_intermediate(files_option) for files_option in files_options
        ]

        if non_existing_files := [src for src, _, _ in src_dst_cloud_intermediate_triplets if not os.path.exists(src)]:
            for x in non_existing_files:
                print(f'did not exist {x} {os.path.exists(x)} {list(os.listdir(os.path.dirname(x)))}')
            non_existing_files_str = '- ' + '\n- '.join(non_existing_files)
            raise HailctlBatchSubmitError(f'Some --files did not exist:\n{non_existing_files_str}', 1)

        for _, dest, cloud_intermediate in src_dst_cloud_intermediate_triplets:
            in_file = b.read_input(cloud_intermediate)
            j.command(f'mkdir -p {shq(os.path.dirname(dest))}; ln -s {shq(in_file)} {shq(dest)}')

        if not os.path.exists(script):
            raise HailctlBatchSubmitError(f'Script file does not exist: {script}', 1)
        script_src, _, script_cloud_file = parse_files_option_to_src_dest_and_cloud_intermediate(script)

        if os.path.exists(user_config_path):
            user_config_src, _, user_config_cloud_file = parse_files_option_to_src_dest_and_cloud_intermediate(
                user_config_path
            )

        local_files_to_cloud_files = [
            {'from': src, 'to': cloud_intermediate}
            for src, _, cloud_intermediate in src_dst_cloud_intermediate_triplets
        ]
        await copy_from_dict(files=local_files_to_cloud_files)
        await copy_from_dict(
            files=[
                {'from': script_src, 'to': script_cloud_file},
                {'from': user_config_src, 'to': user_config_cloud_file},
            ]
        )

        script_file = b.read_input(script_cloud_file)
        config_file = b.read_input(user_config_cloud_file)

        j.env('HAIL_QUERY_BACKEND', 'batch')

        command = 'python3' if script.endswith('.py') else 'bash'
        script_arguments = " ".join(shq(x) for x in arguments)

        j.command('mkdir -p $HOME/.config/hail')
        j.command(f'ln -s {shq(config_file)} $HOME/.config/hail/config.ini')
        j.command(f'mkdir -p {shq(real_absolute_cwd())}')
        j.command(f'cd {shq(real_absolute_cwd())}')
        j.command(f'{command} {shq(script_file)} {script_arguments}')
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
