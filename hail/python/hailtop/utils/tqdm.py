from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.auto import tqdm as tqdm_auto

TQDM_DEFAULT_DISABLE = False if tqdm_auto == tqdm_notebook else None


def tqdm(*args, disable=TQDM_DEFAULT_DISABLE, **kwargs):
    return tqdm_auto(*args, disable=disable, **kwargs)
