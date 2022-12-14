import asyncio
import orjson
import os
import sys
from shlex import quote as shq

import hailtop.batch as hb
import hailtop.batch_client.client as bc
from hailtop import pip_version
from hailtop.aiotools.copy import copy_from_dict
from hailtop.config import get_remote_tmpdir, get_user_config_path, get_deploy_config
from hailtop.utils import secret_alnum_string, unpack_comma_delimited_inputs

HAIL_GENETICS_HAIL_IMAGE = os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}')


def init_parser(parser):
    parser.add_argument('--name', type=str, default='', help='Batch name')
    parser.add_argument('--image-name', type=str, required=False,
                        help='Name for Docker image. Defaults to hailgenetics/hail')
    parser.add_argument('--files', nargs='+', action='append', default=[],
                        help='Comma-separated list of files or directories to add to the working directory of job')
    parser.add_argument('-o', type=str, default='text', choices=['text', 'json'])
    parser.add_argument('script', type=str, help='Path to script')
    parser.add_argument('arguments', nargs='*', help='Arguments to script')


async def async_main(args):
    script = args.script
    files = unpack_comma_delimited_inputs(args.files)
    user_config = get_user_config_path()
    quiet = args.o != 'text'

    remote_tmpdir = get_remote_tmpdir('hailctl batch submit')
    tmpdir_path_prefix = secret_alnum_string()

    def cloud_prefix(path):
        return f'{remote_tmpdir}/{tmpdir_path_prefix}/{path}'

    b = hb.Batch(name=args.name, backend=hb.ServiceBackend())
    j = b.new_bash_job()
    j.image(args.image_name or HAIL_GENETICS_HAIL_IMAGE)

    rel_file_paths = [os.path.relpath(file) for file in files]
    local_files_to_cloud_files = [{'from': local, 'to': cloud_prefix(local)} for local in rel_file_paths]
    await copy_from_dict(files=[
        {'from': script, 'to': cloud_prefix(script)},
        {'from': str(user_config), 'to': cloud_prefix(user_config)},
        *local_files_to_cloud_files,
    ])
    for file in local_files_to_cloud_files:
        local_file = file['from']
        cloud_file = file['to']
        in_file = b.read_input(cloud_file)
        j.command(f'ln -s {in_file} {local_file}')

    script_file = b.read_input(cloud_prefix(script))
    config_file = b.read_input(cloud_prefix(user_config))
    j.command(f'mkdir -p $HOME/.config/hail && ln -s {config_file} $HOME/.config/hail/config.ini')

    j.env('HAIL_QUERY_BACKEND', 'batch')

    command = 'python3' if script.endswith('.py') else 'bash'
    script_arguments = " ".join(shq(x) for x in args.arguments)
    j.command(f'{command} {script_file} {script_arguments}')
    batch_handle: bc.Batch = b.run(wait=False, disable_progress_bar=quiet)  # type: ignore

    if args.o == 'text':
        deploy_config = get_deploy_config()
        url = deploy_config.external_url('batch', f'/batches/{batch_handle.id}/jobs/1')
        print(f'Submitted batch {batch_handle.id}, see {url}')
    else:
        assert args.o == 'json'
        print(orjson.dumps({'id': batch_handle.id}).decode('utf-8'))


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    if pass_through_args:
        print(f'Unrecognized arguments: {" ".join(pass_through_args)}')
        sys.exit(1)
    asyncio.run(async_main(args))
