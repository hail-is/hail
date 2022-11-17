import asyncio

import hailtop.batch as hb
import hailtop.batch_client.client as bc
from hailtop import pip_version
from hailtop.aiotools.copy import copy_from_dict
from hailtop.config import get_remote_tmpdir, get_user_config_path, get_deploy_config
from hailtop.utils import unpack_comma_delimited_inputs


def init_parser(parser):
    parser.add_argument('script', type=str, help='Path to script')
    parser.add_argument('--name', type=str, default='', help='Batch name')
    parser.add_argument('--image-name', type=str, required=False,
                        help='Name for Docker image. Defaults to hailgenetics/hail')
    parser.add_argument('--files', nargs='+', action='append', default=[],
                        help='Comma-separated list of files or directories to add to the working directory of job')


async def async_main(args):
    script = args.script
    files = unpack_comma_delimited_inputs(args.files)
    user_config = get_user_config_path()

    remote_tmpdir = get_remote_tmpdir('hailctl batch submit')

    def cloud_prefix(path):
        return f'{remote_tmpdir}/{path}'

    b = hb.Batch(name=args.name, backend=hb.ServiceBackend())
    j = b.new_bash_job()
    j.image(args.image_name or f'hailgenetics/hail:{pip_version()}')

    local_files_to_cloud_files = [{'from': local, 'to': cloud_prefix(local)} for local in files]
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

    command = 'python3' if script.endswith('.py') else 'bash'
    j.command(f'{command} {script_file}')
    batch_handle: bc.Batch = b.run(wait=False)  # type: ignore

    deploy_config = get_deploy_config()
    url = deploy_config.external_url('batch', f'/batches/{batch_handle.id}/jobs/1')
    print(f'Submitted batch {batch_handle.id}, see {url}')


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    asyncio.run(async_main(args))
