from typing import List


def public_gcr_images(project: str) -> List[str]:
    # the worker cannot import batch_configuration because it does not have all the environment
    # variables
    return [f'gcr.io/{project}/{name}' for name in ('query', 'hail', 'python-dill')]
