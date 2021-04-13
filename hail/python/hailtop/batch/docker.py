import shutil
import sys
import os
from typing import Optional, List
from collections import namedtuple

from ..utils import secret_alnum_string, sync_check_shell_output


def build_python_image(fullname: str,
                       requirements: Optional[List[str]] = None,
                       version: Optional[str] = None,
                       _tmp_dir: str = '/tmp') -> str:
    """
    Build a new Python image with dill and the specified pip packages installed.

    Notes
    -----

    This function is used to build Python images for :class:`.PythonJob`.

    Examples
    --------

    >>> image = build_python_image('gcr.io/hail-vdc/batch-python',
    ...                            requirements=['pandas']) # doctest: +SKIP
    >>> b = Batch(default_python_image=image) # doctest: +SKIP

    Parameters
    ----------
    fullname:
        Full name of where to build the image including any repository prefix and tags
        if desired (default tag is `latest`).
    requirements:
        List of pip packages to install.
    version:
        String in the format of `major_version.minor_version` (ex: `3.7`). Defaults to
        current version of Python that is running.
    _tmp_dir:
        Location to place local temporary files used while building the image.

    Returns
    -------
    Full name where built image is located.
    """
    if version is None:
        version = sys.version_info
    else:
        Version = namedtuple('Version', ['major', 'minor'])
        version = version.split('.')
        version = Version(int(version[0]), int(version[1]))

    if version.major != 3 or version.minor not in (6, 7, 8):
        raise ValueError(
            f'Python versions other than 3.6, 3.7, or 3.8 (you are using {version}) are not supported')

    base_image = f'hailgenetics/python-dill:{version.major}.{version.minor}-slim'

    docker_path = f'{_tmp_dir}/{secret_alnum_string(6)}/docker'
    try:
        print(f'building docker image {fullname}')

        shutil.rmtree(docker_path, ignore_errors=True)
        os.makedirs(docker_path)

        if requirements:
            python_requirements_file = f'{docker_path}/requirements.txt'
            with open(python_requirements_file, 'w') as f:
                f.write('\n'.join(requirements) + '\n')

            with open(f'{docker_path}/Dockerfile', 'w') as f:
                f.write(f'''
FROM {base_image}

COPY requirements.txt .

RUN pip install --upgrade --no-cache-dir -r requirements.txt && \
    python3 -m pip check
''')

            sync_check_shell_output(f'docker build -t {fullname} {docker_path}')
            print(f'finished building image {fullname}')
        else:
            sync_check_shell_output(f'docker pull {base_image}')
            sync_check_shell_output(f'docker tag {base_image} {fullname}')
            print(f'finished pulling image {fullname}')

        image_split = fullname.rsplit('/', 1)
        if len(image_split) == 2:
            sync_check_shell_output(f'docker push {fullname}')
            print(f'finished pushing image {fullname}')
    finally:
        shutil.rmtree(docker_path, ignore_errors=True)

    return fullname
