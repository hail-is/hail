import asyncio

try:
    asyncio.get_running_loop()
    import nest_asyncio
    nest_asyncio.apply()
except RuntimeError as err:
    assert err.args[0] == "no running event loop"
    asyncio.set_event_loop(asyncio.new_event_loop())

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
