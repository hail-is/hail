"""A finished job's retried cost and projected non-preemptible cost, computed from its attempts' billing rows.

Pure: the job API handler fetches the rows and this module does the arithmetic. Vocabulary follows the Batch
glossary (batch/CONTEXT.md): outcome attempt, retried attempt, retried cost, projected non-preemptible cost.
"""

from typing import Iterable, List, NamedTuple, Optional, Set, Tuple


class AttemptResource(NamedTuple):
    """One resource an attempt was billed for, with the attempt's reason and billed interval."""

    attempt_id: str
    reason: Optional[str]
    start_time: Optional[int]
    rollup_time: Optional[int]
    resource: str
    rate: float
    quantity: int


def _cost(row: AttemptResource) -> float:
    # the billing formula: quantity * GREATEST(COALESCE(rollup_time - start_time, 0), 0) * rate
    if row.start_time is None or row.rollup_time is None:
        return 0.0
    return row.quantity * max(row.rollup_time - row.start_time, 0) * row.rate


def nonpreemptible_product(product: str) -> Optional[str]:
    """The non-preemptible counterpart of a preemptible product, or None if the product's price doesn't depend on preemptibility."""
    segments = product.split('/')
    for i, segment in enumerate(segments):
        if segment == 'preemptible':
            segments[i] = 'nonpreemptible'
            return '/'.join(segments)
        if segment.endswith('-preemptible'):
            segments[i] = segment[: -len('-preemptible')] + '-nonpreemptible'
            return '/'.join(segments)
    return None


def rate_in_effect(resources: Iterable[Tuple[str, float]], product: str, time_msecs: int) -> Optional[float]:
    """The rate of `product` in effect at `time_msecs`, from `(resource, rate)` rows, or None if no version was yet in effect.

    A resource is named `{product}/{version}`. An ingested price's version is its effective-start time in epoch
    milliseconds; a legacy resource's is `1`. Versions are compared as numbers, never as strings.
    """
    best: Optional[Tuple[int, float]] = None
    for resource, rate in resources:
        resource_product, version = resource.rsplit('/', 1)
        if resource_product != product or not version.isdigit():
            continue
        effective_start = int(version)
        if effective_start <= time_msecs and (best is None or effective_start > best[0]):
            best = (effective_start, rate)
    return best[1] if best is not None else None


def _product(resource: str) -> str:
    return resource.rsplit('/', 1)[0]


def retried_attempts_cost(
    attempt_resources: Iterable[AttemptResource], outcome_attempt_id: Optional[str]
) -> Optional[float]:
    """The cost of a finished job's retried attempts: every attempt but its outcome attempt and cancelled attempts.

    None if none of the job's attempts was billed for a preemptible product, i.e. the job didn't run preemptible.
    """
    attempt_resources = list(attempt_resources)
    if not any(nonpreemptible_product(_product(row.resource)) is not None for row in attempt_resources):
        return None
    return sum(
        (_cost(row) for row in attempt_resources if row.attempt_id != outcome_attempt_id and row.reason != 'cancelled'),
        0.0,
    )


def _outcome_attempt_resources(
    attempt_resources: Iterable[AttemptResource], outcome_attempt_id: Optional[str]
) -> List[AttemptResource]:
    if outcome_attempt_id is None:
        return []
    return [row for row in attempt_resources if row.attempt_id == outcome_attempt_id]


def nonpreemptible_counterparts(
    attempt_resources: Iterable[AttemptResource], outcome_attempt_id: Optional[str]
) -> Set[str]:
    """The non-preemptible products whose rates the outcome attempt's projected non-preemptible cost needs."""
    counterparts = (
        nonpreemptible_product(_product(row.resource))
        for row in _outcome_attempt_resources(attempt_resources, outcome_attempt_id)
    )
    return {counterpart for counterpart in counterparts if counterpart is not None}


def projected_nonpreemptible_cost(
    attempt_resources: Iterable[AttemptResource],
    outcome_attempt_id: Optional[str],
    counterpart_resources: Iterable[Tuple[str, float]],
) -> Optional[float]:
    """The outcome attempt's cost at the non-preemptible rates in effect when it started.

    Products whose price doesn't depend on preemptibility keep their billed rate. None if the outcome attempt
    wasn't billed for any preemptible product, or if any counterpart rate can't be found.
    """
    outcome = _outcome_attempt_resources(attempt_resources, outcome_attempt_id)
    counterpart_resources = list(counterpart_resources)
    total = 0.0
    billed_preemptible = False
    for row in outcome:
        counterpart = nonpreemptible_product(_product(row.resource))
        if counterpart is None:
            total += _cost(row)
            continue
        if row.start_time is None:
            return None
        rate = rate_in_effect(counterpart_resources, counterpart, row.start_time)
        if rate is None:
            return None
        total += _cost(row._replace(rate=rate))
        billed_preemptible = True
    return total if billed_preemptible else None
