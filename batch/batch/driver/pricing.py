import abc
from typing import Optional

from hailtop.utils import time_msecs


class Price(abc.ABC):
    region: str
    effective_start_date: int
    effective_end_date: Optional[int]
    time_updated: int

    def is_current_price(self):
        now = time_msecs()
        return now >= self.effective_start_date and (self.effective_end_date is None or now <= self.effective_end_date)

    @property
    def version(self) -> str:
        raise NotImplementedError

    @property
    def product(self):
        raise NotImplementedError

    @property
    def rate(self):
        raise NotImplementedError
