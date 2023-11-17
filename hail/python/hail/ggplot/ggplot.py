from dataclasses import asdict, replace
from textwrap import dedent, indent
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from altair import Chart, LayerChart, X, Y, X2
from pandas import DataFrame

import hail as hl
from hail import MatrixTable, Table
from hail.ggplot.utils import typeguard_dataclass


"""
how are we going to cache stats/any other transformations to the data?
id(data), we can cache all transformations on that with a compound, ordered key of stats
for example, (stat1, stat2, stat3) would be distinct from (stat2, stat3, stat1)
and we can get as much of the beginning of that path as possible from the cache, so if we have (stat1, stat2, stat3) in the cache, and attempt to do (stat1, stat2, stat4, stat3), we can get the first two steps from that cache
and in fact we will need to store each new path as its own thing, so if we do stat1 then stat2 then stat3 then roll it back and do stat1 then stat2 then stat4, we should end up with the following keys in the cache under the key for the original data:
()
(stat1)
(stat1, stat2)
(stat1, stat2, stat3)
(stat1, stat2, stat4)
how does the user interact with the cache? if we undo the addition of a geom or stat, we can roll it back, and if they reapply the same stat, we pull from it
no caching ir, if the user wants to use the caching, they should make a stat subclass
expose a function so they can do that
and then the way to change it within the plot object is to use stat_function or whatever to supply your own thing that returns a table

value should be (hail agg, dataframe)

do this for identity too so we cache the df for it
"""


### types ###
Data = Union[Table, MatrixTable]


@typeguard_dataclass
class Mapping:
    x: Optional[str]
    y: Optional[str]
    # TODO add the rest of the supported aesthetic names
    color: Optional[str]


Geom = Literal["bar", "line", "circle"]
Stat = Literal["identity", "bin"]


@typeguard_dataclass
class Layer:
    mapping: Mapping
    data: Optional[Data]
    geom: Optional[Geom]
    stat: Stat
    # FIXME if there's only one type per param name we can make this a typeddict
    params: Dict[str, Any]


@typeguard_dataclass
class Plot:
    data: Optional[Data]
    mapping: Mapping
    layers: list[Layer]


### module-level variables ###
_plot_cache: Dict[int, List[Plot]] = {}
_stat_cache: Dict[Tuple[int, ...], Data] = {}


### constructor functions ###
def aes(x: Optional[str] = None, y: Optional[str] = None, color: Optional[str] = None) -> Mapping:
    return Mapping(x, y, color)


def geom_histogram(mapping: Mapping = aes(), data: Optional[Data] = None, bins: int = 30) -> Layer:
    return Layer(mapping, data, "bar", "bin", {"bins": bins})


def geom_line(mapping: Mapping = aes(), data: Optional[Data] = None) -> Layer:
    return Layer(mapping, data, "line", "identity", {})


def geom_point(mapping: Mapping = aes(), data: Optional[Data] = None) -> Layer:
    return Layer(mapping, data, "circle", "identity", {})


def ggplot(data: Optional[Data] = None, mapping: Mapping = aes()) -> Plot:
    global _plot_cache
    new_plot = Plot(data, mapping, [])
    _plot_cache |= {id(new_plot): []}
    return new_plot


### functionality ###
def extend(plot: Plot, other: Any) -> Plot:
    global _plot_cache
    kwargs: Optional[Dict[str, Any]] = None
    if isinstance(other, Mapping):
        kwargs = {
            "mapping": replace(
                plot.mapping,
                **{k: v for k, v in {"x": other.x, "y": other.y, "color": other.color}.items() if v is not None},
            )
        }
    elif isinstance(other, Layer):
        kwargs = {"layers": [*plot.layers, other]}

    if kwargs is None:
        raise ValueError("unsupported addition to plot")

    new_plot = replace(plot, **kwargs)
    _plot_cache |= {id(new_plot): _plot_cache[id(plot)] + [plot]}
    _plot_cache = {k: v for k, v in _plot_cache.items() if k != id(plot)}
    return new_plot


setattr(Plot, "__add__", extend)


_altair_configure_mark_keys = {"color"}
_altair_encode_keys = {"x": X, "x2": X2, "y": Y}


def show(plot: Plot) -> Union[Chart, LayerChart]:
    global _stat_cache
    base_chart = None
    for layer in plot.layers:
        mapping_dict = {}
        for mapping in [plot.mapping, layer.mapping]:
            mapping_dict = {**mapping_dict, **{k: v for k, v in asdict(mapping).items() if v is not None}}
        # TODO should we break the stat stuff out to its own function?
        kwargs = {"x": {}, "x2": {}, "y": {}}
        cached = _stat_cache.get((id(plot.data), layer.stat), None)
        if cached is not None:
            data, df = cached
        elif layer.stat == "identity":
            data = plot.data
            df = data.to_pandas()
        elif layer.stat == "bin":
            # TODO add caching
            x = mapping_dict.get("x", None)
            if x is None:
                raise ValueError("x must be supplied for stat bin")
            data = plot.data.aggregate(
                hl.agg.hist(
                    plot.data[x],
                    plot.data.aggregate(hl.agg.min(plot.data[x])),
                    plot.data.aggregate(hl.agg.max(plot.data[x])),
                    layer.params["bins"],
                )
            )
            df = DataFrame(
                [
                    {x: data["bin_edges"][i], "x2": data["bin_edges"][i + 1], "y": data["bin_freq"][i]}
                    for i in range(len(data["bin_freq"]))
                ]
            )
            kwargs["x"] = {"bin": "binned"}
            mapping_dict["x2"] = "x2"
            mapping_dict["y"] = "y"
        else:
            raise ValueError("unknown stat")
        _stat_cache |= {(id(plot.data), layer.stat): (data, df)}
        chart = Chart(df)
        if layer.geom is not None:
            chart = getattr(chart, f"mark_{layer.geom}")(
                **{k: v for k, v in mapping_dict.items() if k in _altair_configure_mark_keys}
            )
        chart = chart.encode(
            **{k: _altair_encode_keys[k](v, **kwargs[k]) for k, v in mapping_dict.items() if k in _altair_encode_keys}
        )
        base_chart = chart if base_chart is None else base_chart + chart
    return base_chart


def undo(plot: Plot, *, depth: int = 1) -> Plot:
    global _plot_cache
    old_plot = _plot_cache[id(plot)][0 - depth]
    _plot_cache |= {id(old_plot): _plot_cache[id(plot)][: 0 - depth]}
    _plot_cache = {k: v for k, v in _plot_cache.items() if k != id(plot)}
    return old_plot


def plot_to_string(plot: Plot) -> str:
    return dedent(
        f"""\
        Plot(
            data = {plot.data},
            mapping = {indent_tail(str(plot.mapping), 3)},
            layers = {indent_tail(str(plot.layers), 3)},
        )"""
    )


def indent_tail(string: str, indent_level: int = 1) -> str:
    return "".join(
        [
            indent(part, "    " * indent_level) if index == 2 else part
            for index, part in enumerate(string.partition("\n"))
        ]
    )


setattr(Plot, "__str__", plot_to_string)


def mapping_to_string(mapping: Mapping) -> str:
    return dedent(
        f"""\
        Mapping(
            x = {mapping.x},
            y = {mapping.y},
        )"""
    )


setattr(Mapping, "__str__", mapping_to_string)
