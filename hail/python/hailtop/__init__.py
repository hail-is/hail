from hailtop.version import __pip_version__, __version__

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


__all__ = [
    '__pip_version__',
    '__version__',
    'is_notebook',
]
