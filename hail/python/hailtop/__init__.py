_VERSION = None


def version() -> str:
    global _VERSION
    if _VERSION is None:
        import pkg_resources  # pylint: disable=import-outside-toplevel
        _VERSION = pkg_resources.resource_string(__name__, 'hail_version').decode().strip()
    return _VERSION


def pip_version() -> str:
    return version().split('-')[0]
