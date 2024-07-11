from collections.abc import Generator
from typing import Any, Dict, List, Optional

import hail as hl
from hail.ggplot import GGPlot, aes, geom_line, geom_point, geom_vline, ggplot, ggtitle


def __agg_names(ht: hl.Table) -> List[str]:
    return ht.aggregate(hl.array(hl.agg.collect_as_set(ht.name)))


def plot_trial_against_time(
    ht: hl.Table,
    names: Optional[List[str]] = None,
    first_stable_index: Optional[Dict[str, int]] = None,
) -> Generator[GGPlot, Any, Any]:
    for name in names or __agg_names(ht):
        k = ht.filter(ht.name == name)
        k = k.explode(k.instances, name='__instance')
        k = k.select(**k.__instance)
        k = k.explode(k.trials, name='trial')

        plot = ggplot(k, aes(x=k.trial.iteration, y=k.trial.time, color=hl.str(k.instance)))
        plot += geom_line()
        plot += ggtitle(name)

        if first_stable_index is not None:
            plot += geom_vline(xintercept=first_stable_index[name])

        yield plot


def plot_mean_time_per_instance(
    ht: hl.Table,
    names: Optional[List[str]] = None,
) -> Generator[GGPlot, Any, Any]:
    for name in names or __agg_names(ht):
        k = ht.filter(ht.name == name)
        k = k.explode(k.instances, name='__instance')
        k = k.select(**k.__instance)
        k = k.annotate(s=k.trials.aggregate(lambda t: hl.agg.stats(t.time)))
        yield (ggplot(k, aes(x=k.instance, y=k.s.mean)) + geom_point() + ggtitle(name))
