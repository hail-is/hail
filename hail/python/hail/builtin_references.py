import threading

BUILTIN_REFERENCE_DOWNLOAD_LOCKS = {
    'GRCh37': threading.Lock(),
    'GRCh38': threading.Lock(),
    'GRCm38': threading.Lock(),
    'CanFam3': threading.Lock(),
}
BUILTIN_REFERENCES = tuple(BUILTIN_REFERENCE_DOWNLOAD_LOCKS.keys())
