from typing import List
from hailtop.batch import hail_genetics_images

def public_gcr_images(project: str) -> List[str]:
    # the worker cannot import batch_configuration because it does not have all the environment
    # variables
    return [f'gcr.io/{project}/{name}' for name in ('query', 'hail', 'python-dill')]
