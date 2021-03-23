import shutil
import os
from typing import Optional, List

from ..utils import secret_alnum_string, sync_check_shell_output


def build_python_image(build_dir: str,
                       base_image: str,
                       dest_image: str,
                       python_requirements: Optional[List[str]] = None,
                       verbose: bool = True):
    docker_path = f'{build_dir}/{secret_alnum_string(6)}/docker'
    try:
        print(f'building docker image {dest_image}')

        shutil.rmtree(docker_path, ignore_errors=True)
        os.makedirs(docker_path)

        if python_requirements:
            python_requirements = python_requirements or []
            python_requirements_file = f'{docker_path}/requirements.txt'
            with open(python_requirements_file, 'w') as f:
                f.write('\n'.join(python_requirements) + '\n')

            with open(f'{docker_path}/Dockerfile', 'w') as f:
                f.write(f'''
FROM {base_image}

COPY requirements.txt .

RUN pip install --upgrade --no-cache-dir -r requirements.txt && \
    python3 -m pip check
''')

            sync_check_shell_output(f'docker build -t {dest_image} {docker_path}', echo=verbose)
            print(f'finished building image {dest_image}')
        else:
            sync_check_shell_output(f'docker pull {base_image}', echo=verbose)
            sync_check_shell_output(f'docker tag {base_image} {dest_image}')
            print(f'finished pulling image {dest_image}')

        image_split = dest_image.rsplit('/', 1)
        if len(image_split) == 2:
            sync_check_shell_output(f'docker push {dest_image}', echo=verbose)
            print(f'finished pushing image {dest_image}')
    finally:
        shutil.rmtree(docker_path, ignore_errors=True)
