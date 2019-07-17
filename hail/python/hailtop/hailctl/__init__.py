def version() -> str:
    import pkg_resources
    return pkg_resources.resource_string(__name__, 'hail_version').decode().strip()


__all__ = [
    'version'
]
