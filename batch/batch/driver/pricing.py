import abc
from typing import Optional

from hailtop.utils import time_msecs


class Price(abc.ABC):
    def __init__(self, *, region: str, effective_start_date: int, effective_end_date: Optional[int], sku: str):
        self.region = region
        self.effective_start_date = effective_start_date
        self.effective_end_date = effective_end_date
        self.sku = sku

    def is_current_price(self):
        now = time_msecs()
        return now >= self.effective_start_date and (self.effective_end_date is None or now <= self.effective_end_date)

    @property
    def version(self) -> str:
        return str(self.effective_start_date)

    @property
    @abc.abstractmethod
    def product(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def rate(self) -> float:
        pass
