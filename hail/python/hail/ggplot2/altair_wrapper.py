from typing import Any, Union

from altair import Chart
from pandas import DataFrame

import hail
from hail import MatrixTable, Table

Data = Union[Table, MatrixTable]


class ChartWrapper:
    def __init__(self, data: Data, *args, **kwargs) -> None:
        self.chart_args = args
        self.chart_kwargs = kwargs
        self.data = data

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "data":
            self.update_data()

    def update_data(self) -> None:
        self.cache = {}
        self.chart = Chart(self.data.to_pandas(), *self.chart_args, **self.chart_kwargs)

    def histogram(self, x: str, bins: int = 30) -> None:
        if (aggregated := self.cache.get("histogram", None)) is None:
            self.cache["histogram"] = (
                aggregated := self.data.aggregate(
                    hail.agg.hist(
                        self.data[x],
                        self.data.aggregate(hail.agg.min(self.data[x])),
                        self.data.aggregate(hail.agg.max(self.data[x])),
                        bins,
                    )
                )
            )
        self.chart = Chart(
            DataFrame([
                {"x": aggregated["bin_edges"][i], "x2": aggregated["bin_edges"][i + 1], "y": aggregated["bin_freq"][i]}
                for i in range(len(aggregated["bin_freq"]))
            ]),
            *self.chart_args,
            **self.chart_kwargs,
        )
