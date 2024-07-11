from os import cpu_count
from typing import Callable

import hail as hl


def cv(trial: hl.StructExpression) -> hl.Float64Expression:
    """coefficient of variation"""
    return hl.bind(lambda s: s.stdev / s.mean, hl.agg.stats(trial))


def variability(ht: hl.Table) -> hl.Table:
    """Compute benchmark total and per-trial variability"""
    return ht.select(
        total=ht.instances.aggregate(
            lambda trials: hl.agg.explode(cv, trials),
        ),
        trials=(
            ht.instances.map(lambda trials: trials.aggregate(cv)).aggregate(
                lambda cvs: hl.agg.stats(cvs).select('mean', 'stdev')
            )
        ),
    )


def boostrap_confidence_interval(
    statistic: Callable[[hl.Table], hl.NumericExpression],
    ht: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f'Confidence must fall within interval (0, 1), got {confidence}.')

    endpoints = (lower := (1 - confidence) / 2, 1 - lower)

    ht = ht.annotate(__bootstrap=hl.range(n_bootstrap_iterations))
    ht = ht.explode('__bootstrap')
    ht = ht.select(__bootstrap=statistic(ht))
    ht = ht.group_by(*ht.key).aggregate(__results=hl.agg.collect(ht.__bootstrap))

    # not obvious why, but this checkpoint adds a 4x performance improvement
    ht = ht.checkpoint(hl.utils.new_temp_file())

    return ht.select(
        **hl.bind(
            lambda results, len: hl.bind(
                lambda lo, hi: hl.struct(
                    ci=hl.interval(lo, hi, includes_end=True),
                    radius=hl.rbind((hi + lo) / 2, lambda mid: (hi - mid) / mid),
                ),
                *[results[hl.int(p * len)] for p in endpoints],
            ),
            hl.sorted(ht.__results),
            hl.int(hl.len(ht.__results)),
        ),
    )


def __randomize_with_replacement(xs: hl.ArrayExpression) -> hl.ArrayExpression:
    return hl.bind(
        lambda xs: hl.bind(
            lambda len: hl.map(
                lambda idx: xs[idx],
                hl.repeat(lambda: hl.rand_int32(len), len),
            ),
            hl.len(xs),
        ),
        xs,
    )


def __agg_randomized_mean(instances: hl.ArrayStructExpression) -> hl.NumericExpression:
    return (
        __randomize_with_replacement(instances)
        .map(__randomize_with_replacement)
        .aggregate(lambda trials: hl.agg.explode(hl.agg.mean, trials))
    )


def bootstrap_mean_confidence_interval(
    ht: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Approximate confidence interval of mean execution time across all trails
    of a benchmark via bootstrap simulations as proposed by Laaber et al.
    """
    return boostrap_confidence_interval(
        statistic=lambda ht: __agg_randomized_mean(ht.instances),
        ht=ht,
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )


def bootstrap_ib_difference_confidence_interval(
    ht: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Approximate confidence interval of difference in mean execution time of
    control and test groups of benchmark trials on different instances via
    bootstrap simulations.
    """
    return boostrap_confidence_interval(
        statistic=lambda ht: __agg_randomized_mean(ht.control) / __agg_randomized_mean(ht.test),
        ht=ht,
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )


def bootstrap_tb_difference_confidence_interval(
    ht: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Approximate confidence interval of difference in mean execution time of
    control and test groups of benchmark trials on same instances via
    bootstrap simulations.
    """

    def agg_mean(trials: hl.ArrayNumericExpression) -> hl.NumericExpression:
        return hl.agg.explode(hl.agg.mean, __randomize_with_replacement(trials))

    def ratio_of_means(ht: hl.Table) -> hl.NumericExpression:
        return __randomize_with_replacement(ht.instances).aggregate(
            lambda instance: agg_mean(instance.control) / agg_mean(instance.test)
        )

    return boostrap_confidence_interval(
        statistic=ratio_of_means,
        ht=ht,
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )


def overlapping_confidence_interval_test(
    control: hl.Table,
    test: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Test for performance changes by comparing overlapping confidence intervals
    of mean execution time between a control and test set of benchmark timings
    """
    control = bootstrap_mean_confidence_interval(control, n_bootstrap_iterations, confidence)
    test = bootstrap_mean_confidence_interval(test, n_bootstrap_iterations, confidence)
    return control.select(overlaps=test[control.key].ci.overlaps(control.ci))


def analyze_benchmarks(
    control: hl.Table,
    test: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Example:

    import hail as hl

    from benchmark.tools.impex import import_timings
    from benchmark.tools.statistics import analyse_benchmarks
    from pathlib import Path

    tables = [
        import_timings(Path('data') / f'{version}.jsonl')
        for version in ['0.2.132', '0.2.133']
    ]

    control, test = [
        table
        .select(instances=table.instances.trials.time)
        ._key_by_assert_sorted(*table.key.drop('version'))
        for table in tables
    ]

    results = analyze_benchmarks(
        control,
        test,
        n_bootstrap_iterations=10_000,
        confidence=.95,
    )

    results.show()
    """

    results = overlapping_confidence_interval_test(
        control,
        test,
        n_bootstrap_iterations,
        confidence,
    )

    diffs = bootstrap_ib_difference_confidence_interval(
        control.select(
            control=control.instances,
            test=test[control.key].instances,
        ),
        n_bootstrap_iterations,
        confidence,
    )

    return hl.Table.parallelize(
        key=control.key.dtype.fields,
        rows=hl.sorted(
            results.select(changed=~results.overlaps, relative_change=diffs[results.key].ci).collect(_localize=False),
            key=lambda r: (r.path, r.name),
        ),
    )


def __select_disjoint(n: hl.Int32Expression, xs: hl.ArrayExpression) -> hl.TupleExpression:
    return hl.bind(
        lambda n, xs: hl.bind(
            lambda len: hl.case()
            .when(n <= 2 * len, hl.bind(lambda ys: hl.tuple([ys[:n], ys[n : 2 * n]]), hl.shuffle(xs)))
            .or_error("split position '" + hl.str(n) + "' exceeds twice array length '" + hl.str(len) + "'."),
            hl.len(xs),
        ),
        n,
        xs,
    )


def __ibs(
    ht: hl.Table,
    ninstances: hl.Int32Expression,
    ntrials: hl.Int32Expression,
) -> hl.Table:
    return ht.select(
        **dict(
            zip(
                ('control', 'test'),
                __select_disjoint(ninstances, ht.instances.map(lambda trials: hl.shuffle(trials)[:ntrials])),
            )
        ),
    )


def __tbs(
    ht: hl.Table,
    ninstances: hl.Int32Expression,
    ntrials: hl.Int32Expression,
) -> hl.Table:
    return ht.select(
        **hl.shuffle(ht.instances)[:ninstances].aggregate(
            lambda i: hl.struct(**{
                group: hl.agg.collect(trials)
                for group, trials in zip(
                    ('control', 'test'),
                    __select_disjoint(ntrials, i),
                )
            })
        )
    )


def __sel(strategy: Callable, ht: hl.Table, n_experiments: int) -> hl.Table:
    ht = ht.annotate(experiment=hl.range(n_experiments)).explode('experiment')
    ht = ht._key_by_assert_sorted(*ht.key, 'experiment')
    return strategy(ht, ht.ninstances, ht.ntrials)


def __scale(
    instances: hl.ArrayExpression,
    factor: hl.Float64Expression,
) -> hl.ArrayExpression:
    return instances.map(lambda instance: instance.map(lambda trial: trial * factor))


def __extend_key(ht: hl.Table, **kwargs) -> hl.Table:
    return ht.annotate(**kwargs)._key_by_assert_sorted(*ht.key, *kwargs)


def laaber_mds(
    ht: hl.Table,
    n_bootstrap_iterations: int = 1_000,
    n_experiments: int = 100,
    confidence: float = 0.95,
) -> hl.Table:
    """Minimal detectable slowdown as described in Laaber et al"""

    s = (
        ht.annotate(
            slowdown=[1 + x / 10 for x in range(0, 6)],
            ninstances=[5, 10, 15, 20, 25],
            ntrials=[5, 10, 15, 20, 25],
        )
        .explode('slowdown')
        .explode('ninstances')
        .explode('ntrials')
        ._key_by_assert_sorted(*ht.key, 'slowdown', 'ninstances', 'ntrials')
        .repartition(cpu_count() * 16)
    )

    ib = __sel(__ibs, __extend_key(s, strategy='ibs'), n_experiments)
    tb = __sel(__tbs, __extend_key(s, strategy='tbs'), n_experiments)
    s = ib.union(tb)
    s = s.annotate(test=__scale(s.test, s.slowdown))
    s = s.checkpoint(hl.utils.new_temp_file())

    mds = overlapping_confidence_interval_test(
        s.select(instances=s.control),
        s.select(instances=s.test),
        n_bootstrap_iterations,
        confidence,
    )

    return mds.group_by(*mds.key.drop('strategy', 'experiment')).aggregate(
        **hl.bind(
            lambda groups: hl.struct(**{k: groups[k] for k in ['ibs', 'tbs']}),
            hl.agg.group_by(mds.strategy, hl.agg.fraction(~mds.overlaps)),
        )
    )


def schultz_mds(
    ht: hl.Table,
    n_bootstrap_iterations: int = 1_000,
    n_experiments: int = 100,
    confidence: float = 0.95,
) -> hl.Table:
    s = (
        ht.annotate(
            slowdown=[1 + x / 10 for x in range(0, 6)],
            ninstances=[5, 10, 15, 20, 25],
            ntrials=[5, 10, 15, 20, 25],
        )
        .explode('slowdown')
        .explode('ninstances')
        .explode('ntrials')
        ._key_by_assert_sorted(*ht.key, 'slowdown', 'ninstances', 'ntrials')
        .repartition(cpu_count() * 7)
    )

    ib = __sel(__ibs, s, n_experiments)
    ib = bootstrap_ib_difference_confidence_interval(
        ib.annotate(test=__scale(ib.test, ib.slowdown)),
        n_bootstrap_iterations,
        confidence,
    )

    ib = ib.group_by(*ib.key.drop('experiment')).aggregate(rate=hl.agg.fraction(~ib.ci.contains(1.0)))

    tb = __sel(__tbs, s, n_experiments)
    tb = bootstrap_tb_difference_confidence_interval(
        tb.select(
            instances=hl.map(
                lambda control, test: hl.struct(control=control, test=test),
                tb.control,
                __scale(tb.test, tb.slowdown),
            ),
        ),
        n_bootstrap_iterations,
        confidence,
    )

    tb = tb.group_by(*tb.key.drop('experiment')).aggregate(rate=hl.agg.fraction(~tb.ci.contains(1.0)))

    return ib.select(ibs=ib.rate, tbs=tb[ib.key].rate)
