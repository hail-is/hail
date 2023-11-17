from dataclasses import asdict, replace
from textwrap import dedent, indent
from typing import Any, Dict, List, Literal, Optional, Set, Union, Tuple

from altair import Chart, LayerChart, X, Y, X2
from pandas import DataFrame

import hail as hl
from hail import MatrixTable, Table
from hail.expr import Expression, literal
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
"""


Data = Union[Table, MatrixTable]
Geom = Literal["bar", "line", "circle"]
Stat = Literal["identity", "bin"]


@typeguard_dataclass
class Mapping:
    x: Optional[str]
    y: Optional[str]
    # FIXME add the rest of the supported aesthetic names
    color: Optional[str] = None


@typeguard_dataclass
class Layer:
    mapping: Mapping
    data: Optional[Data]
    geom: Optional[Geom]
    stat: Optional[Stat]
    # FIXME if there's only one type per param name we can make this a typeddict
    params: Dict[str, Any]


@typeguard_dataclass
class Plot:
    data: Optional[Data]
    mapping: Mapping
    layers: list[Layer]


_plot_cache: Dict[int, List[Plot]] = {}
_stat_cache: Dict[Tuple[int, ...], Data] = {}


def aes(x: Optional[str] = None, y: Optional[str] = None, **rest: Optional[str]) -> Mapping:
    return Mapping(**{"x": x, "y": y}, **rest)


def to_expressions(mapping: Dict[str, Any]) -> Dict[str, Optional[Expression]]:
    return {
        k: v if (v is None and k in {"x", "y"}) or isinstance(v, Expression) else literal(v) for k, v in mapping.items()
    }


def ggplot(data: Optional[Data] = None, mapping: Mapping = aes()) -> Plot:
    new_plot = Plot(data, mapping, [])
    global _plot_cache
    _plot_cache |= {id(new_plot): []}
    return new_plot


def extend(plot: Plot, other: Any) -> Plot:
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

    if kwargs is not None:
        new_plot = replace(plot, **kwargs)
        global _plot_cache
        _plot_cache |= {id(new_plot): _plot_cache[id(plot)] + [plot]}
        _plot_cache = {k: v for k, v in _plot_cache.items() if k != id(plot)}
        return new_plot
    return plot


setattr(Plot, "__add__", extend)


def plot_to_string(plot: Plot) -> str:
    return dedent(
        f"""\
        Plot(
            data = {plot.data},
            mapping = {indent_tail(str(plot.mapping), 3)},
            layers = {indent_tail(str(plot.layers), 3)},
        )"""
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


def undo(plot: Plot, *, depth: int = 1) -> Plot:
    global _plot_cache
    old_plot = _plot_cache[id(plot)][0 - depth]
    _plot_cache |= {id(old_plot): _plot_cache[id(plot)][: 0 - depth]}
    _plot_cache = {k: v for k, v in _plot_cache.items() if k != id(plot)}
    return old_plot


ALTAIR_CONFIGURE_MARK_KEYS = {"color"}
ALTAIR_ENCODE_KEYS = {"x": X, "x2": X2, "y": Y}


def show(plot: Plot) -> Union[Chart, LayerChart]:
    base_chart = None
    df = plot.data.to_pandas()
    for layer in plot.layers:
        _df = df
        kwargs = {"x": {}, "x2": {}, "y": {}}
        mapping_dict = to_nonmissing_dict(plot.mapping, layer.mapping)
        if layer.geom is None:
            raise ValueError("layer must have a geom")
        if layer.stat is not None:
            if layer.stat == "bin":
                # FIXME don't hardcode the field names
                # FIXME why is there a gap in the bins when we have 30?
                agg = plot.data.aggregate(
                    hl.agg.hist(
                        plot.data["idx_2"],
                        plot.data.aggregate(hl.agg.min(plot.data["idx_2"])),
                        plot.data.aggregate(hl.agg.max(plot.data["idx_2"])),
                        layer.params["bins"],
                    )
                )
                _df = DataFrame(
                    [
                        {"x": agg["bin_edges"][i], "x2": agg["bin_edges"][i + 1], "y": agg["bin_freq"][i]}
                        for i in range(len(agg["bin_freq"]))
                    ]
                )
                kwargs["x"] = {"bin": "binned"}
                mapping_dict["x2"] = "x2"
            elif layer.stat != "identity":
                raise ValueError("unknown stat")
        chart = getattr(Chart(_df), f"mark_{layer.geom}")(
            **filter_to_keys(mapping_dict, ALTAIR_CONFIGURE_MARK_KEYS)
        ).encode(
            **{k: ALTAIR_ENCODE_KEYS[k](v, **kwargs[k]) for k, v in mapping_dict.items() if k in ALTAIR_ENCODE_KEYS}
        )
        base_chart = chart if base_chart is None else base_chart + chart
    return base_chart


def filter_to_keys(mapping_dict: Dict[str, Optional[str]], keys: Set[str]) -> Dict[str, Optional[str]]:
    return {k: v for k, v in mapping_dict.items() if k in keys}


def to_nonmissing_dict(*mappings: Mapping) -> Dict[str, Optional[str]]:
    acc = {}
    for mapping in mappings:
        acc = {**acc, **{k: v for k, v in asdict(mapping).items() if v is not None}}
    return acc


def indent_tail(string: str, indent_level: int = 1) -> str:
    return "".join(
        [
            indent(part, "    " * indent_level) if index == 2 else part
            for index, part in enumerate(string.partition("\n"))
        ]
    )


def geom_histogram(mapping: Mapping = aes(), data: Optional[Data] = None, bins: int = 30) -> Layer:
    return Layer(mapping, data, "bar", "bin", {"bins": bins})


def geom_line(mapping: Mapping = aes(), data: Optional[Data] = None) -> Layer:
    return Layer(mapping, data, "line", None, {})


def geom_point(mapping: Mapping = aes(), data: Optional[Data] = None) -> Layer:
    return Layer(mapping, data, "circle", None, {})
