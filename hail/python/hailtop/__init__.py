import nest_asyncio
nest_asyncio.apply()

_VERSION = None


def version() -> str:
    global _VERSION
    if _VERSION is None:
        import pkg_resources  # pylint: disable=import-outside-toplevel
        _VERSION = pkg_resources.resource_string(__name__, 'hail_version').decode().strip()
    return _VERSION


def pip_version() -> str:
    return version().split('-')[0]


IS_NOTEBOOK = None


def is_notebook() -> bool:
    global IS_NOTEBOOK
    if IS_NOTEBOOK is None:
        try:
            from IPython.core.getipython import get_ipython  # pylint: disable=import-outside-toplevel
            IS_NOTEBOOK = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except (NameError, ModuleNotFoundError):
            IS_NOTEBOOK = False
    return IS_NOTEBOOK
