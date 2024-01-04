import orjson
import os
from shlex import quote as shq
from hailtop import pip_version


async def submit(name, image_name, files, output, script, arguments):
    import hailtop.batch as hb  # pylint: disable=import-outside-toplevel
    from hailtop.aiotools.copy import copy_from_dict  # pylint: disable=import-outside-toplevel
    from hailtop.config import (
        get_remote_tmpdir,
        get_user_config_path,
        get_deploy_config,
    )  # pylint: disable=import-outside-toplevel
    from hailtop.utils import (
        secret_alnum_string,
        unpack_comma_delimited_inputs,
    )  # pylint: disable=import-outside-toplevel

    files = unpack_comma_delimited_inputs(files)
    user_config = get_user_config_path()
    quiet = output != 'text'

    remote_tmpdir = get_remote_tmpdir('hailctl batch submit')
    tmpdir_path_prefix = secret_alnum_string()

    def cloud_prefix(path):
        return f'{remote_tmpdir}/{tmpdir_path_prefix}/{path}'

    backend = hb.ServiceBackend()
    b = hb.Batch(name=name, backend=backend)
    j = b.new_bash_job()
    j.image(image_name or os.environ.get('HAIL_GENETICS_HAIL_IMAGE', f'hailgenetics/hail:{pip_version()}'))

    rel_file_paths = [os.path.relpath(file) for file in files]
    local_files_to_cloud_files = [{'from': local, 'to': cloud_prefix(local)} for local in rel_file_paths]
    await copy_from_dict(
        files=[
            {'from': script, 'to': cloud_prefix(script)},
            {'from': str(user_config), 'to': cloud_prefix(user_config)},
            *local_files_to_cloud_files,
        ]
    )
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

    await backend.async_close()
