import asyncio
import sys
from asyncio import Event, Semaphore
from types import TracebackType
from typing import Awaitable, Callable, List, Literal, Optional, Tuple, Type, TypeVar, Union, cast

from sortedcontainers import SortedKeyList

from hailtop.utils.utils import WithoutSemaphore

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


async def weighted_bounded_gather2_raise_exceptions(
    sema: WeightedSemaphore, pfs: List[Callable[[], Awaitable[T]]], weights: List[int], cancel_on_error: bool = False
) -> List[T]:
    """Similar to `bounded_gather2_raise_exceptions`, but uses a
    weighted semaphore to schedule tasks with different weights.
    """

    async def run_with_sema(pf: Callable[[], Awaitable[T]], weight: int):
        async with sema.acquire_manager(weight):
            return await pf()

    tasks = [asyncio.create_task(run_with_sema(pf, weight)) for pf, weight in zip(pfs, weights)]

    if not cancel_on_error:
        async with WithoutSemaphore(sema):
            return await asyncio.gather(*tasks)

    try:
        async with WithoutSemaphore(sema):
            return await asyncio.gather(*tasks)
    finally:
        _, exc, _ = sys.exc_info()
        if exc is not None:
            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        raise exc
                else:
                    task.cancel()
            if tasks:
                async with WithoutSemaphore(sema):
                    await asyncio.wait(tasks)


async def weighted_bounded_gather2_return_exceptions(
    sema: WeightedSemaphore, pfs: List[Callable[[], Awaitable[T]]], weights: List[int]
) -> List[Union[Tuple[T, None], Tuple[None, Optional[BaseException]]]]:
    """Similar to `bounded_gather2_return_exceptions`, but uses a
    weighted semaphore to schedule tasks with different weights.
    """

    async def run_with_sema_return_exceptions(pf: Callable[[], Awaitable[T]], weight: int):
        try:
            async with sema.acquire_manager(weight):
                return (await pf(), None)
        except:
            _, exc, _ = sys.exc_info()
            return (None, exc)

    tasks = [asyncio.create_task(run_with_sema_return_exceptions(pf, weight)) for pf, weight in zip(pfs, weights)]
    async with WithoutSemaphore(sema):
        return await asyncio.gather(*tasks)


async def weighted_bounded_gather2(
    sema: WeightedSemaphore,
    pfs: List[Callable[[], Awaitable[T]]],
    weights: Optional[List[int]] = None,
    return_exceptions: bool = False,
    cancel_on_error: bool = False,
) -> List[T]:
    if not weights:
        weights = [1] * len(pfs)
    if return_exceptions:
        if cancel_on_error:
            raise ValueError('cannot request return_exceptions and cancel_on_error')
        return await weighted_bounded_gather2_return_exceptions(sema, pfs, weights)  # type: ignore
    return await weighted_bounded_gather2_raise_exceptions(sema, pfs, weights, cancel_on_error=cancel_on_error)
