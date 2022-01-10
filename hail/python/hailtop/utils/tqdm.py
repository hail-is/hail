TQDM_DEFAULT_DISABLE = object()


def tqdm(*args, disable=TQDM_DEFAULT_DISABLE, **kwargs):
    from tqdm.notebook import tqdm as tqdm_notebook  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm as tqdm_auto  # pylint: disable=import-outside-toplevel
    # To tqdm_notebook, None means do not display. To standard tqdm, None means
    # display only when connected to a TTY.
    if disable is TQDM_DEFAULT_DISABLE:
        disable = False if tqdm_auto == tqdm_notebook else None
    return tqdm_auto(*args, disable=disable, **kwargs)
