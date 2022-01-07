def version() -> str:
    import pkg_resources  # pylint: disable=import-outside-toplevel
    return pkg_resources.resource_string(__name__, 'hail_version').decode().strip()


def pip_version() -> str:
    return version().split('-')[0]
