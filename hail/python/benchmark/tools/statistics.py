"""
Provides many of the statistical methods used to analyse hail's cloud
benchmark results. These methods are based off [1].

[Note on terminology]

In [1], Laaber et al. are concerned with reliably detecting slowdowns in
micro-benchmarks run in cloud environments. They use the following
terminology:
- Iteration: single timed invocation of a micro-benchmark
-     Trail: fixed number of iterations run consecutively
-  Instance: virtual machine of a certain instance-type provided by a cloud vendor

In their execution strategy, Laaber et al. create N INSTANCES of various
instance-types and schedule 10 randomised TRIALS of 50 ITERATIONS of each
benchmark.

Hail's benchmarks in hail/python/benchmark/hail are "macro"-benchmarks and
can take some tens of seconds per ITERATION. Our execution strategy is slightly
different too in that we use hail batch jobs as our INSTANCES (see
[Note on batch workers]). I think it's important that readers can recognise the
methods discussed in [1] when reading this code. Therefore, I've adopted the
following terminology:
- Iteration: single timed invocation of a benchmark
-  Instance: a batch job executing a fixed number of consecutive iterations

[Note on batch jobs]
Jobs run on standard preemptable workers with 2 vCPUs and 8 GB of memory.
As of writing, we do not use job-private workers and so don't control what else
runs on those workers. We only benchmark on Google Cloud.

[References]

[1] Laaber et al., Software Microbenchmarking in the Cloud. How Bad is it Really?
https://dl.acm.org/doi/10.1007/s10664-019-09681-1
"""

from typing import Callable

import hail as hl


def agg_cv(iteration: hl.NumericExpression) -> hl.Float64Expression:
    """aggregate the coefficient of variation across iterations"""
    return hl.bind(lambda s: s.stdev / s.mean, hl.agg.stats(iteration))


def variability(ht: hl.Table) -> hl.StructExpression:
    """Compute benchmark total and per-instance variability"""
    return hl.struct(
        total=ht.instances.aggregate(
            lambda iterations: hl.agg.explode(agg_cv, iterations),
        ),
        instances=(
            ht.instances.map(lambda iterations: iterations.aggregate(agg_cv)).aggregate(
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
    """
    Bootstrap estimates of confidence intervals of each `statistic` with `n_bootstrap_iterations`.
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f'`confidence` must fall within interval (0, 1), got {confidence}.')

    endpoints = (lower := (1 - confidence) / 2, 1 - lower)
    endpoints = [hl.int(p * n_bootstrap_iterations) for p in endpoints]

    def confidence_interval(statistic: hl.ArrayExpression) -> hl.StructExpression:
        return hl.rbind(
            hl.sorted(statistic),
            lambda results: hl.rbind(
                *[results[p] for p in endpoints],
                lambda lo, hi: hl.struct(
                    ci=hl.interval(lo, hi, includes_end=True),
                    radius=hl.rbind((hi + lo) / 2, lambda mid: (hi - mid) / mid),
                ),
            ),
        )

    return ht.select(
        **hl.rbind(
            hl.range(n_bootstrap_iterations)
            .map(lambda _: statistics(ht))
            .aggregate(lambda stats: hl.struct(**{k: hl.agg.collect(v) for k, v in stats.items()})),
            lambda stats: hl.struct(**{k: confidence_interval(vs) for k, vs in stats.items()}),
        ),
    )


def __randomize_with_replacement(xs: hl.ArrayExpression) -> hl.ArrayExpression:
    return hl.rbind(
        xs,
        lambda xs: hl.rbind(
            hl.len(xs),
            lambda len: hl.map(
                lambda idx: xs[idx],
                hl.repeat(lambda: hl.rand_int32(len), len),
            ),
        ),
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
    of mean execution time between a `control` and `test` set of benchmark timings.
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
    return hl.rbind(
        n,
        xs,
        lambda n, xs: hl.rbind(
            hl.len(xs),
            lambda len: hl.case()
            .when(2 * n <= len, hl.rbind(hl.shuffle(xs), lambda ys: hl.tuple([ys[:n], ys[n : 2 * n]])))
            .or_error("split position '" + hl.str(n) + "' exceeds half array length '" + hl.str(len) + "'."),
        ),
    )


# [Note on selection strategies]
#
# We want to compare what effect, if any, running on the same machine vs
# running on different machines. Laaber et al. propose two ways of selecting
# data to simulate this:
#
# "Instance Based Selection" (ibs) is a selection strategy that simulates
# running control and test benchmarks on different machines.
#
# "Trial Based Selection" is a selection strategy that simulates running
# control and test benchmarks on the same machines. The name "trial" here
# is a carry-over from Laaber et al. but applies to iterations in
# our case.
#
# These selection strategies sample the original dataset into two distinct
# `control` and `test` datasets that are used to a/b test simulated slowdowns.


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


def __annotate_explode_configurations(
    ht: hl.Table,
    slowdown: list[float] | None = None,
    ninstances: list[int] | None = None,
    niterations: list[int] | None = None,
) -> hl.Table:
    # Using a slowdown of 1 helps rule out those configurations that yield a
    # high rate of false positives.
    slowdown = slowdown or [1, 1.1, 1.2, 1.3, 1.4, 1.5]
    ninstances = ninstances or [5, 10, 15, 20, 25]
    niterations = niterations or [5, 10, 15, 20, 25]

    # [Note on parallelism]
    #
    # The bootstrap simulations are quite computationally intensive. We can use
    # hail's partition-level parallelism to increase throughput, especially
    # when using the 'batch' backend. This `pfactor` aims to make each
    # configuration of (slowdown, ninstances, niterations) run in parallel.
    pfactor = len(slowdown) * len(ninstances) * len(niterations)

    return (
        ht.annotate(slowdown=slowdown, ninstances=ninstances, niterations=niterations)
        .explode('slowdown')
        .explode('ninstances')
        .explode('niterations')
        ._key_by_assert_sorted(*ht.key, 'slowdown', 'ninstances', 'niterations')
        .repartition(ht.n_partitions() * pfactor)
    )


# [Note on implementation]
#
# In general, I found it conceptually simpler to maintain the following
# invariants:
# - Tables are distinctly keyed
# - When considering rows as `[key, value]` pairs
#   - callers must save any required values in the key
#   - callees must return tables with their key unaltered
#
# This allowed me to write functions that freely modify the row value. The key
# acts like a stack; contextual information is pushed onto the stack (such as
# experiment number or simulated slowdown) and the table is re-keyed.
#
# This may not be the most efficient thing to do. I thought that it would help
# expose parallelism across bootstrap simulations (see [Note on parallelism]).


def laaber_mds(
    ht: hl.Table,
    n_bootstrap_iterations: int = 1_000,
    n_experiments: int = 100,
    confidence: float = 0.95,
) -> hl.Table:
    """
    Minimal detectable slowdown as described in Laaber et al.

    Estimates slowdown detection by considering the fraction of experiments
    whose bootstrapped confidence intervals of mean execution time do not
    overlap for varying configuration and selection strategies.

    Default values taken from [1].
    """

    ht = __annotate_explode_configurations(ht)

    ib = __sel(__ibs, __extend_key(ht, strategy='ibs'), n_experiments)
    tb = __sel(__tbs, __extend_key(ht, strategy='tbs'), n_experiments)

    ht = ib.union(tb)
    ht = ht.annotate(test=__scale(ht.test, ht.slowdown))

    mds = overlapping_confidence_interval_test(
        ht.checkpoint(hl.utils.new_temp_file()),
        n_bootstrap_iterations,
        confidence,
    )

    return mds.group_by(*mds.key.drop('strategy', 'experiment')).aggregate(
        **hl.rbind(
            hl.agg.group_by(mds.strategy, hl.agg.fraction(~mds.overlaps)),
            lambda groups: hl.struct(**{k: groups[k] for k in ['ibs', 'tbs']}),
        )
    )


def schultz_mds(
    ht: hl.Table,
    n_bootstrap_iterations: int = 1_000,
    n_experiments: int = 100,
    confidence: float = 0.95,
) -> hl.Table:
    """
    Patrick's minimal detectable slowdown.

    Estimates slowdown detection by considering the fraction of experiments
    whose bootstrapped confidence intervals of the ratio of mean execution
    time between control and test benchmarks that do not contain 1 for varying
    configuration and selection strategies.
    """
    ht = __annotate_explode_configurations(ht)

    ib = __sel(__ibs, ht, n_experiments)

    ib = boostrap_confidence_interval(
        ht=ib.annotate(test=__scale(ib.test, ib.slowdown)),
        statistics=lambda t: hl.struct(diff=__agg_randomized_mean(t.control) / __agg_randomized_mean(t.test)),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )

    tb = __sel(__tbs, ht, n_experiments)

    # Recall that in trial-based selection, both `control` and `test` iterations
    # are run on the same machine. Here we can aggregate the per-instance ratio
    # of mean execution time as opposed to instance-based selection where we can
    # only consider the ratio across all instances.
    def agg_ratio_of_means(ht: hl.Table) -> hl.NumericExpression:
        def agg_mean(iterations: hl.ArrayNumericExpression) -> hl.NumericExpression:
            return hl.agg.explode(hl.agg.mean, __randomize_with_replacement(iterations))

        return __randomize_with_replacement(ht.instances).aggregate(
            lambda instance: agg_mean(instance.control) / agg_mean(instance.test)
        )

    tb = boostrap_confidence_interval(
        ht=tb.select(
            instances=hl.map(
                lambda control, test: hl.struct(control=control, test=test),
                tb.control,
                __scale(tb.test, tb.slowdown),
            ),
        ),
        statistics=lambda t: hl.struct(diff=agg_ratio_of_means(t)),
        n_bootstrap_iterations=n_bootstrap_iterations,
        confidence=confidence,
    )

    ib, tb = [
        t.group_by(*t.key.drop('experiment')).aggregate(rate=hl.agg.fraction(~t.diff.ci.contains(1.0)))
        for t in [ib, tb]
    ]

    return ib.select(ibs=ib.rate, tbs=tb[ib.key].rate)
