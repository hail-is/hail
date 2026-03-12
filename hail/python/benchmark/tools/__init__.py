import logging
from typing import Any, List, Optional

from hail.expr import ArrayStructExpression
from hail.expr import enumerate as hl_enumerate


def prune(kvs: dict) -> dict:
    return {k: v for k, v in kvs.items() if v is not None}


def select(keys: List[str], **kwargs) -> List[Optional[Any]]:
    return [kwargs.get(k) for k in keys]


def annotate_index(arr: ArrayStructExpression) -> ArrayStructExpression:
    return hl_enumerate(arr).map(lambda t: t[1].annotate(idx=t[0]))


def init_logging(file=None):
    logging.basicConfig(format="%(asctime)-15s: %(levelname)s: %(message)s", level=logging.INFO, filename=file)
