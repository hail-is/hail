import abc


class AsyncStream(abc.ABC):
    def __init__(self):
        self._closed = False
        self._waited_closed = False

    def readable(self) -> bool:
        return False

    async def read(self, n: int = -1) -> bytes:
        raise NotImplementedError

    def writable(self) -> bool:
        return False

    async def write(self, b: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True

    @abc.abstractmethod
    async def _wait_closed(self) -> None:
        pass

    async def wait_closed(self) -> None:
        self._closed = True
        if not self._waited_closed:
            try:
                await self._wait_closed()
            finally:
                self._waited_closed = True

    @property
    def closed(self) -> None:
        return self._closed

    async def __aenter__(self) -> 'AsyncStream[_T]':
        return self

    async def __aexit__(
            self, exc_type: Optional[Type[BaseException]] = None,
            exc_value: Optional[BaseException] = None,
            exc_traceback: Optional[TracebackType] = None) -> None:
        await self.wait_closed()
