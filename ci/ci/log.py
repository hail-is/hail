import logging

fmt = logging.Formatter(
    # NB: no space after levename because WARNING is so long
    '%(levelname)s\t| %(asctime)s \t| %(filename)s \t| %(funcName)s:%(lineno)d | '
    '%(message)s')

fh = logging.FileHandler('ci.log')
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)

log = logging.getLogger('ci')
log.setLevel(logging.INFO)

logging.basicConfig(
    handlers=[fh, ch],
    level=logging.INFO)
