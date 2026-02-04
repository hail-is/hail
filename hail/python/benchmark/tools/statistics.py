from typing import Callable

import hail as hl


def cv(iteration: hl.StructExpression) -> hl.Float64Expression:
    """coefficient of variation"""
    return hl.bind(lambda s: s.stdev / s.mean, hl.agg.stats(iteration))


def variability(ht: hl.Table) -> hl.StructExpression:
    """Compute benchmark total and per-trial variability"""
    return hl.struct(
        total=ht.instances.aggregate(
            lambda iterations: hl.agg.explode(cv, iterations),
        ),
        iterations=(
            ht.instances.map(lambda iterations: iterations.aggregate(cv)).aggregate(
                lambda cvs: hl.agg.stats(cvs).select('mean', 'stdev')
            )
        ),
    )


def boostrap_confidence_interval(
    ht: hl.Table,
    statistics: Callable[[hl.Table], hl.StructExpression],
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f'Confidence must fall within interval (0, 1), got {confidence}.')

    endpoints = (lower := (1 - confidence) / 2, 1 - lower)
    endpoints = [hl.int(p * n_bootstrap_iterations) for p in endpoints]

    def confidence_interval(statistic: hl.ArrayExpression) -> hl.StructExpression:
        return hl.bind(
            lambda results: hl.bind(
                lambda lo, hi: hl.struct(
                    ci=hl.interval(lo, hi, includes_end=True),
                    radius=hl.rbind((hi + lo) / 2, lambda mid: (hi - mid) / mid),
                ),
                *[results[p] for p in endpoints],
            ),
            hl.sorted(statistic),
        )

    ht = ht.select(
        __statistics=(
            hl.range(n_bootstrap_iterations)
            .map(lambda _: statistics(ht))
            .aggregate(lambda stats: hl.struct(**{k: hl.agg.collect(v) for k, v in stats.items()}))
        ),
    )

    return ht.select(
        **hl.bind(
            lambda stats: hl.struct(**{k: confidence_interval(v) for k, v in stats.items()}),
            ht.__statistics,
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
        .aggregate(lambda iterations: hl.agg.explode(hl.agg.mean, iterations))
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
        ht=ht,
        statistics=lambda t: hl.struct(mean=__agg_randomized_mean(t.instances)),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )


def overlapping_confidence_interval_test(
    ht: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Test for performance changes by comparing overlapping confidence intervals
    of mean execution time between a control and test set of benchmark timings
    """
    result = boostrap_confidence_interval(
        ht=ht,
        statistics=lambda t: hl.struct(
            control=__agg_randomized_mean(t.control),
            test=__agg_randomized_mean(t.test),
        ),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )
    return result.select(overlaps=result.control.ci.overlaps(result.test.ci))


def analyze_benchmarks(
    ht: hl.Table,
    n_bootstrap_iterations: int,
    confidence: float,
) -> hl.Table:
    """
    Example:

    import hail as hl

    from benchmark.tools.impex import import_benchmarks
    from benchmark.tools.statistics import analyse_benchmarks
    from pathlib import Path

    tables = [
        import_benchmarks(Path('data') / f'{version}.jsonl')
        for version in ['0.2.132', '0.2.133']
    ]

    control, test = [
        table
        .select(instances=table.instances.iterations.time)
        ._key_by_assert_sorted(*table.key.drop('version'))
        for table in tables
    ]

    results = analyze_benchmarks(
        control.select(control=control.instances, test=test[control.key].instances),
        n_bootstrap_iterations=10_000,
        confidence=.95,
    )

    results.show()
    """

    results = boostrap_confidence_interval(
        ht=ht,
        statistics=lambda t: hl.bind(
            lambda a, b: hl.struct(
                control=a,
                test=b,
                relative_change=a / b,
            ),
            __agg_randomized_mean(t.control),
            __agg_randomized_mean(t.test),
        ),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )

    return results.select(
        changed=~results.control.ci.overlaps(results.test.ci),
        relative_chage=results.relative_change,
    )


def __select_disjoint(n: hl.Int32Expression, xs: hl.ArrayExpression) -> hl.TupleExpression:
    return hl.bind(
        lambda n, xs: hl.bind(
            lambda len: hl.case()
            .when(2 * n <= len, hl.bind(lambda ys: hl.tuple([ys[:n], ys[n : 2 * n]]), hl.shuffle(xs)))
            .or_error("split position '" + hl.str(n) + "' exceeds half array length '" + hl.str(len) + "'."),
            hl.len(xs),
        ),
        n,
        xs,
    )


def __ibs(
    ht: hl.Table,
    ninstances: hl.Int32Expression,
    niterations: hl.Int32Expression,
) -> hl.Table:
    return ht.select(
        **dict(
            zip(
                ('control', 'test'),
                __select_disjoint(
                    ninstances,
                    ht.instances.map(lambda iterations: hl.shuffle(iterations)[:niterations]),
                ),
            )
        ),
    )


def __tbs(
    ht: hl.Table,
    ninstances: hl.Int32Expression,
    niterations: hl.Int32Expression,
) -> hl.Table:
    return ht.select(
        **hl.shuffle(ht.instances)[:ninstances].aggregate(
            lambda i: hl.struct(**{
                group: hl.agg.collect(iterations)
                for group, iterations in zip(
                    ('control', 'test'),
                    __select_disjoint(niterations, i),
                )
            })
        )
    )


def __sel(strategy: Callable, ht: hl.Table, n_experiments: int) -> hl.Table:
    ht = ht.annotate(experiment=hl.range(n_experiments)).explode('experiment')
    ht = ht._key_by_assert_sorted(*ht.key, 'experiment')
    return strategy(ht, ht.ninstances, ht.niterations)


def __scale(
    instances: hl.ArrayExpression,
    factor: hl.Float64Expression,
) -> hl.ArrayExpression:
    return hl.map(
        lambda instance: hl.map(
            lambda iteration: iteration * factor,
            instance,
        ),
        instances,
    )


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
            slowdown=[1, 1.2, 1.3, 1.4, 1.5],
            ninstances=[5, 10, 15, 20, 25],
            niterations=[5, 10, 15, 20, 25],
        )
        .explode('slowdown')
        .explode('ninstances')
        .explode('niterations')
        ._key_by_assert_sorted(*ht.key, 'slowdown', 'ninstances', 'niterations')
        .repartition(ht.n_partitions() * 6**3)
    )

    ib = __sel(__ibs, __extend_key(s, strategy='ibs'), n_experiments)
    tb = __sel(__tbs, __extend_key(s, strategy='tbs'), n_experiments)

    s = ib.union(tb)
    s = s.annotate(test=__scale(s.test, s.slowdown))

    mds = overlapping_confidence_interval_test(
        s.checkpoint(hl.utils.new_temp_file()),
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
            slowdown=[1, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5],
            ninstances=[5, 10, 15, 20, 25, 30],
            niterations=[5, 10, 15, 20, 25, 30],
        )
        .explode('slowdown')
        .explode('ninstances')
        .explode('niterations')
        ._key_by_assert_sorted(*ht.key, 'slowdown', 'ninstances', 'niterations')
        .repartition(ht.n_partitions() * 6**3)
    )

    ib = __sel(__ibs, s, n_experiments)
    ib = boostrap_confidence_interval(
        ht=ib.annotate(test=__scale(ib.test, ib.slowdown)),
        statistics=lambda t: hl.struct(diff=__agg_randomized_mean(t.control) / __agg_randomized_mean(t.test)),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )

    ib = ib.group_by(*ib.key.drop('experiment')).aggregate(rate=hl.agg.fraction(~ib.diff.ci.contains(1.0)))

    def ratio_of_means(ht: hl.Table) -> hl.NumericExpression:
        def agg_mean(iterations: hl.ArrayNumericExpression) -> hl.NumericExpression:
            return hl.agg.explode(hl.agg.mean, __randomize_with_replacement(iterations))

        return __randomize_with_replacement(ht.instances).aggregate(
            lambda instance: agg_mean(instance.control) / agg_mean(instance.test)
        )

    tb = __sel(__tbs, s, n_experiments)
    tb = boostrap_confidence_interval(
        ht=tb.select(
            instances=hl.map(
                lambda control, test: hl.struct(control=control, test=test),
                tb.control,
                __scale(tb.test, tb.slowdown),
            ),
        ),
        statistics=lambda t: hl.struct(ratio=ratio_of_means(t)),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )

    tb = tb.group_by(*tb.key.drop('experiment')).aggregate(rate=hl.agg.fraction(~tb.ratio.ci.contains(1.0)))

    return ib.select(ibs=ib.rate, tbs=tb[ib.key].rate)
