from typing import List


def publicly_available_images(docker_prefix: str) -> List[str]:
    # the worker cannot import batch_configuration because it does not have all the environment
    # variables
    return [
        f'{docker_prefix}/{name}'
        # FIXME: remove raw python-dill when we stop supporting clients 0.2.79 and before
        for name in ('query', 'hailgenetics/hail', 'hailgenetics/genetics', 'python-dill', 'hailgenetics/python-dill', 'batch-worker')
    ]
