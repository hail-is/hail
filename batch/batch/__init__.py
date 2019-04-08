import logging
import sys

from . import api, client, aioclient


def make_logger():
    fmt = logging.Formatter(
        # NB: no space after levename because WARNING is so long
        '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
        '%(message)s')

    file_handler = logging.FileHandler('batch.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    log = logging.getLogger('batch')
    log.setLevel(logging.INFO)

    logging.basicConfig(handlers=[file_handler, stream_handler], level=logging.INFO)

    return log


log = make_logger()


def run_once(target, *args, **kwargs):
    try:
        log.info(f'run_forever: {target.__name__}')
        target(*args, **kwargs)
        log.info(f'run_forever: {target.__name__} returned')
    except Exception:  # pylint: disable=W0703
        log.error(f'run_forever: {target.__name__} caught_exception: ', exc_info=sys.exc_info())


__all__ = [
    'client',
    'aioclient',
    'api',
    'run_once'
]
