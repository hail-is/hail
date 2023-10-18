import orjson
import os
from shlex import quote as shq
from hailtop import pip_version


async def submit(name, image_name, files, mounts, output, script, arguments):
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

    script = os.path.expanduser(script)
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

    rel_file_paths = []
    for file in files:
        rel_path = os.path.relpath(file)
        if rel_path.startswith('..'):
            raise ValueError(f'File {file} is located in a parent of the current directory. Use the --mounts option with a mount point specified instead.')
        rel_file_paths.append(rel_path)

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

    mount_files_to_cloud_files = []
    for _input in mounts:
        if _input.startswith('file://'):
            input = _input[7:]
        else:
            input = _input
        if ':' not in input:
            raise ValueError('Must specify mount point separated by a colon (ex: foo.py:/foo/)')

        source, mount = input.split(':')
        source = os.path.expanduser(source)

        if not mount.startswith('/'):
            raise ValueError(f'Mount point must start with a "/". Found {mount} for source {source}.')

        if os.path.isfile(source):
            mount = mount.rstrip('/') + '/'
            dest = mount + os.path.basename(source)
        else:
            dest = mount

        cloud_file = cloud_prefix(dest.lstrip('/'))

        mount_files_to_cloud_files.append({'from': source, 'to': cloud_file})

        in_file = b.read_input(cloud_file)
        j.command(f'mkdir -p {os.path.dirname(dest)}; ln -s {in_file} {dest}')

    await copy_from_dict(files=[*mount_files_to_cloud_files])

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
