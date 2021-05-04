from typing import List


def publicly_available_images(docker_prefix: str) -> List[str]:
    # the worker cannot import batch_configuration because it does not have all the environment
    # variables
    return [
        f'{docker_prefix}/{name}'
        for name in ('query', 'hail', 'python-dill', 'batch-worker')
    ]
