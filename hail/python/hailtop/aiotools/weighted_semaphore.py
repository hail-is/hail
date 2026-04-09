from asyncio import Event, Semaphore
from types import TracebackType
from typing import Literal, Optional, Type, TypeVar, cast

from sortedcontainers import SortedKeyList

T = TypeVar('T')  # pylint: disable=invalid-name


class _AcquireManager:
    def __init__(self, ws: 'WeightedSemaphore', n: int):
        self._ws = ws
        self._n = n

    async def __aenter__(self) -> '_AcquireManager':
        await self._ws.acquire(self._n)
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self._ws.release(self._n)


class WeightedSemaphore(Semaphore):
    def __init__(self, value: int):
        super().__init__(value)
        self.max = value
        self._value = value
        self.events = SortedKeyList(key=lambda x: x[0])

    def locked(self):
        return self._value == 0 and (any(not event.is_set() for n, event in (self.events or ())))

    def release(self, n: int = 1) -> None:
        self._value += n
        self._wake_up_next()

    def _wake_up_next(self):
        while self.events:
            _n, _event = self.events[0]
            # cast to int / Event:
            _n = cast(int, _n)
            _event = cast(Event, _event)

            if self._value >= _n:
                self.events.pop(0)
                self._value -= _n
                _event.set()
            else:
                break

    def acquire_manager(self, n: int) -> _AcquireManager:
        return _AcquireManager(self, n)

    async def acquire(self, n: int = 1) -> Literal[True]:
        assert n <= self.max
        if self._value >= n:
            self._value -= n
            return True

        event = Event()
        self.events.add((n, event))
        await event.wait()
        return True
