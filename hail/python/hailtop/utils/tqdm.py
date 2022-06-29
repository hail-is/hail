from typing import Union, Optional
from enum import Enum


class TqdmDisableOption(Enum):
    default = 0


def tqdm(*args, disable: Optional[Union[TqdmDisableOption, bool]] = TqdmDisableOption.default, **kwargs):
    from tqdm.notebook import tqdm as tqdm_notebook  # pylint: disable=import-outside-toplevel
    from tqdm.auto import tqdm as tqdm_auto  # pylint: disable=import-outside-toplevel
    # To tqdm_notebook, None means do not display. To standard tqdm, None means
    # display only when connected to a TTY.
    if disable == TqdmDisableOption.default:
        disable = False if tqdm_auto == tqdm_notebook else None
    return tqdm_auto(*args, disable=disable, **kwargs)
