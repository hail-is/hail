import orjson
import os
import re
from shlex import quote as shq
from hailtop import pip_version
from typing import Tuple


FILE_REGEX = re.compile(r'(?P<src>[^:]+)(:(?P<dest>.+))?')


async def submit(name, image_name, files, output, script, arguments):
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

    files = unpack_comma_delimited_inputs(files)
    user_config = str(get_user_config_path())

    quiet = output != 'text'

    remote_tmpdir = get_remote_tmpdir('hailctl batch submit')
    tmpdir_path_prefix = secret_alnum_string()

    def cloud_prefix(path):
        path = path.lstrip('/')
        return f'{remote_tmpdir}/{tmpdir_path_prefix}/{path}'

    def file_input_to_src_dest(file: str) -> Tuple[str, str, str]:
        match = FILE_REGEX.match(file)
        if match is None:
            raise ValueError(f'invalid file specification {file}. Must have the form "src" or "src:dest"')

        result = match.groupdict()

        src = result.get('src')
        if src is None:
            raise ValueError(f'invalid file specification {file}. Must have a "src" defined.')
        src = os.path.abspath(os.path.expanduser(src))
        src = src.rstrip('/')

        dest = result.get('dest')
        if dest is not None:
            dest = os.path.abspath(os.path.expanduser(dest))
        else:
            dest = os.getcwd()

        cloud_file = cloud_prefix(src)

        return (src, dest, cloud_file)

    backend = hb.ServiceBackend()
    b = hb.Batch(name=name, backend=backend)
    j = b.new_bash_job()
    j.image(image_name or os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}'))

    local_files_to_cloud_files = []

    for file in files:
        src, dest, cloud_file = file_input_to_src_dest(file)
        local_files_to_cloud_files.append({'from': src, 'to': cloud_file})
        in_file = b.read_input(cloud_file)
        j.command(f'mkdir -p {os.path.dirname(dest)}; ln -s {in_file} {dest}')

    script_src, _, script_cloud_file = file_input_to_src_dest(script)
    user_config_src, _, user_config_cloud_file = file_input_to_src_dest(user_config)

    assert False, str(local_files_to_cloud_files)

    await copy_from_dict(files=[
        {'from': script_src, 'to': script_cloud_file},
        {'from': user_config_src, 'to': user_config_cloud_file},
        *local_files_to_cloud_files])

    script_file = b.read_input(script_cloud_file)
    config_file = b.read_input(user_config_cloud_file)

    j.env('HAIL_QUERY_BACKEND', 'batch')

    command = 'python3' if script.endswith('.py') else 'bash'
    script_arguments = " ".join(shq(x) for x in arguments)

    j.command(f'mkdir -p $HOME/.config/hail && ln -s {config_file} $HOME/.config/hail/config.ini')
    j.command(f'cd {os.getcwd()}')
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

    await backend.async_close()
