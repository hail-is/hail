from collections.abc import Generator
from typing import Any, Dict, List

import hail as hl
from benchmark.tools import annotate_index
from hail.ggplot import GGPlot, aes, geom_line, geom_point, geom_vline, ggplot, ggtitle


def __agg_names(ht: hl.Table) -> List[str]:
    return ht.aggregate(hl.array(hl.agg.collect_as_set(ht.name)))


def plot_iteration_against_time(
    ht: hl.Table,
    names: List[str] | None = None,
    first_stable_index: Dict[str, int] | None = None,
) -> Generator[GGPlot, Any, Any]:
    for name in names or __agg_names(ht):
        k = ht.filter(ht.name == name)
        k = k.explode(k.instances, name='__instance')
        k = k.select(**k.__instance)
        k = k.annotate(iterations=annotate_index(k.iterations))
        k = k.explode(k.iterations, name='iteration')

        plot = ggplot(k, aes(x=k.iteration.idx, y=k.iteration.time, color=hl.str(k.job_id)))
        plot += geom_line()
        plot += ggtitle(name)

        if first_stable_index is not None:
            plot += geom_vline(xintercept=first_stable_index.get(name))

        yield plot


def plot_mean_time_per_instance(
    ht: hl.Table,
    names: List[str] | None = None,
) -> Generator[GGPlot, Any, Any]:
    for name in names or __agg_names(ht):
        k = ht.filter(ht.name == name)
        k = k.annotate(instances=annotate_index(k.instances))
        k = k.explode(k.instances, name='__instance')
        k = k.select(**k.__instance)
        k = k.annotate(s=k.iterations.aggregate(lambda t: hl.agg.stats(t.time)))
        yield (ggplot(k, aes(x=k.idx, y=k.s.mean)) + geom_point() + ggtitle(name))
