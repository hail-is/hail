from collections.abc import Generator
from typing import Any, List, Optional

import hail as hl
from hail.ggplot import GGPlot, aes, geom_line, geom_point, ggplot, ggtitle


def __agg_names(ht: hl.Table) -> List[str]:
    return ht.aggregate(hl.array(hl.agg.collect_as_set(ht.name)))


def plot_trial_against_time(
    ht: hl.Table,
    names: Optional[List[str]] = None,
) -> Generator[GGPlot, Any, Any]:
    for name in names or __agg_names(ht):
        k = ht.filter(ht.name == name)
        k = k.explode(k.instances, name='__instance')
        k = k.select(**k.__instance)
        k = k.explode(k.trials, name='trial')
        yield (
            ggplot(k, aes(x=k.trial.iteration, y=k.trial.time, color=hl.str(k.instance))) + geom_line() + ggtitle(name)
        )


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
