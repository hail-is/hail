import pkg_resources


_VERSION = pkg_resources.resource_string(__name__, 'hail_version').decode().strip()


def pip_version() -> str:
    return _VERSION.split('-')[0]


def version() -> str:
    return _VERSION
