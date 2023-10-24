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


_hail_created_the_main_thread_event_loop = False


def hail_event_loop():
    '''If an event loop exists and Hail did not create it, use nest_asyncio to allow Hail's event
    loops to nest inside it.

    If an event loop exists and Hail did create it, then a developer is trying to use async inside
    sync inside async. That is forbidden.

    If no event loop exists, ask asyncio to get one for us.

    '''
    global _hail_created_the_main_thread_event_loop

    import asyncio  # pylint: disable=import-outside-toplevel
    import nest_asyncio  # pylint: disable=import-outside-toplevel

    try:
        asyncio.get_running_loop()

        if _hail_created_the_main_thread_event_loop:
            raise ValueError(
                'As a matter of Hail team policy, you are not allowed to nest asynchronous Hail code '
                'inside synchronous Hail code.'
            )

        nest_asyncio.apply()
        return asyncio.get_running_loop()
    except RuntimeError as err:
        return asyncio.get_event_loop()
