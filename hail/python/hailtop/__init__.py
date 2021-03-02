import pkg_resources


_VERSION = pkg_resources.resource_string(__name__, 'hail_version').decode().strip()


def version() -> str:
    return _VERSION
