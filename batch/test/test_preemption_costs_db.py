"""A job's preemption costs, read from a seeded batch database."""

import secrets
from typing import Dict, List

import pytest

from batch.front_end.job_preemption_costs import ATTEMPT_RESOURCES_SQL, get_job_preemption_costs

from .db_query_checks import assert_scoped, count_row_reads, explain
from .db_seed import T0, Attempt, Job, Update, seed_batch

# a seed resource whose price doesn't depend on preemptibility
DISK = 'seed/disk/1'
DISK_RATE = 0.00001


class Products:
    """A preemptible compute product and its non-preemptible counterpart, unique to one test.

    ``resources`` is global and the db fixture is shared by the whole session, so each test gets products of its
    own; the counterpart lookup would otherwise find another test's rates.
    """

    def __init__(self):
        tag = secrets.token_hex(4)
        self.preemptible = f'seed/compute/p{tag}-preemptible'
        self.nonpreemptible = f'seed/compute/p{tag}-nonpreemptible'


async def _insert_resources(db, rates: Dict[str, float]):
    await db.execute_many('INSERT INTO resources (resource, rate) VALUES (%s, %s)', list(rates.items()))


async def _seed_job(db, job: Job) -> int:
    seeded = await seed_batch(db, [Update(jobs=[job])])
    return seeded.batch_id


async def _costs(db, batch_id: int, job_id: int = 1):
    record = await db.execute_and_fetchone(
        'SELECT state, attempt_id FROM jobs WHERE batch_id = %s AND job_id = %s', (batch_id, job_id)
    )
    return await get_job_preemption_costs(db, batch_id, job_id, record['state'], record['attempt_id'])


def _preempted(attempt_id: str, start: int, resources: Dict[str, int]) -> Attempt:
    return Attempt(attempt_id, start_time=start, end_time=start + 1_000, reason='preempted', resources=resources)


@pytest.fixture
async def products(db) -> Products:
    p = Products()
    await _insert_resources(db, {f'{p.preemptible}/1': 0.001, f'{p.nonpreemptible}/1': 0.004})
    return p


async def test_succeeded_after_preemptions(db, products):
    compute = {f'{products.preemptible}/1': 2}
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                _preempted('a', T0, compute),
                _preempted('b', T0 + 2_000, compute),
                Attempt('c', start_time=T0 + 4_000, end_time=T0 + 7_000, reason='completed', resources=compute),
            ],
        ),
    )
    costs = await _costs(db, batch_id)
    # two retried attempts of 2 units for 1s at the preemptible rate
    assert costs['retried_attempts_cost'] == pytest.approx(2 * (2 * 1_000 * 0.001))
    # the 3s outcome attempt at the non-preemptible rate
    assert costs['projected_nonpreemptible_cost'] == pytest.approx(2 * 3_000 * 0.004)


async def test_succeeded_first_time(db, products):
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                Attempt(
                    'a',
                    start_time=T0,
                    end_time=T0 + 1_000,
                    reason='completed',
                    resources={f'{products.preemptible}/1': 2},
                )
            ],
        ),
    )
    assert await _costs(db, batch_id) == {
        'retried_attempts_cost': 0.0,
        'projected_nonpreemptible_cost': pytest.approx(2 * 1_000 * 0.004),
    }


async def test_projection_keeps_rates_that_dont_depend_on_preemptibility(db, products):
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                Attempt(
                    'a',
                    start_time=T0,
                    end_time=T0 + 1_000,
                    reason='completed',
                    resources={f'{products.preemptible}/1': 2, DISK: 10},
                )
            ],
        ),
    )
    costs = await _costs(db, batch_id)
    assert costs['projected_nonpreemptible_cost'] == pytest.approx(2 * 1_000 * 0.004 + 10 * 1_000 * DISK_RATE)


@pytest.mark.parametrize('state,reason', [('Failed', 'completed'), ('Error', 'error')])
async def test_unsuccessful_after_preemption_has_retried_cost_but_no_projection(db, products, state, reason):
    compute = {f'{products.preemptible}/1': 2}
    batch_id = await _seed_job(
        db,
        Job(
            state=state,
            attempts=[
                _preempted('a', T0, compute),
                Attempt('b', start_time=T0 + 2_000, end_time=T0 + 3_000, reason=reason, resources=compute),
            ],
        ),
    )
    assert await _costs(db, batch_id) == {
        'retried_attempts_cost': pytest.approx(2 * 1_000 * 0.001),
        'projected_nonpreemptible_cost': None,
    }


async def test_cancelled_after_preemption_excludes_the_cancelled_attempt(db, products):
    compute = {f'{products.preemptible}/1': 2}
    batch_id = await _seed_job(
        db,
        Job(
            state='Cancelled',
            attempts=[
                _preempted('a', T0, compute),
                Attempt('b', start_time=T0 + 2_000, end_time=T0 + 5_000, reason='cancelled', resources=compute),
            ],
        ),
    )
    assert await _costs(db, batch_id) == {
        'retried_attempts_cost': pytest.approx(2 * 1_000 * 0.001),
        'projected_nonpreemptible_cost': None,
    }


async def test_running_job_has_no_costs_yet(db, products):
    compute = {f'{products.preemptible}/1': 2}
    batch_id = await _seed_job(
        db,
        Job(state='Running', attempts=[_preempted('a', T0, compute), Attempt('b', start_time=T0 + 2_000)]),
    )
    assert await _costs(db, batch_id) == {'retried_attempts_cost': None, 'projected_nonpreemptible_cost': None}


async def test_nonpreemptible_job_has_no_costs(db, products):
    compute = {f'{products.nonpreemptible}/1': 2}
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                _preempted('a', T0, compute),
                Attempt('b', start_time=T0 + 2_000, end_time=T0 + 3_000, reason='completed', resources=compute),
            ],
        ),
    )
    assert await _costs(db, batch_id) == {'retried_attempts_cost': None, 'projected_nonpreemptible_cost': None}


async def test_projection_uses_the_rate_in_effect_when_the_outcome_attempt_started(db):
    p = Products()
    start = T0 + 50_000
    await _insert_resources(
        db,
        {
            f'{p.preemptible}/1': 0.001,
            f'{p.nonpreemptible}/1': 0.004,
            f'{p.nonpreemptible}/{start - 1}': 0.006,
            f'{p.nonpreemptible}/{start + 1}': 0.008,
        },
    )
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                Attempt(
                    'a',
                    start_time=start,
                    end_time=start + 1_000,
                    reason='completed',
                    resources={f'{p.preemptible}/1': 2},
                )
            ],
        ),
    )
    costs = await _costs(db, batch_id)
    assert costs['projected_nonpreemptible_cost'] == pytest.approx(2 * 1_000 * 0.006)


@pytest.mark.parametrize(
    'counterpart_versions',
    [
        pytest.param([], id='no counterpart'),
        pytest.param([T0 + 1], id='counterpart only priced after the attempt started'),
    ],
)
async def test_no_projection_without_a_counterpart_rate(db, counterpart_versions: List[int]):
    p = Products()
    await _insert_resources(
        db, {f'{p.preemptible}/1': 0.001, **{f'{p.nonpreemptible}/{v}': 0.004 for v in counterpart_versions}}
    )
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                Attempt(
                    'a', start_time=T0, end_time=T0 + 1_000, reason='completed', resources={f'{p.preemptible}/1': 2}
                )
            ],
        ),
    )
    assert await _costs(db, batch_id) == {'retried_attempts_cost': 0.0, 'projected_nonpreemptible_cost': None}


async def test_attempt_resources_query_reads_only_the_job(db, noise_batch, products):
    plan = await explain(db, ATTEMPT_RESOURCES_SQL, (noise_batch.batch_id, 1))
    assert_scoped(plan)

    resources = {f'{products.preemptible}/1': 2, DISK: 10}
    batch_id = await _seed_job(
        db,
        Job(
            state='Success',
            attempts=[
                _preempted('a', T0, resources),
                _preempted('b', T0 + 2_000, resources),
                Attempt('c', start_time=T0 + 4_000, end_time=T0 + 5_000, reason='completed', resources=resources),
            ],
        ),
    )
    rows, reads = await count_row_reads(db, ATTEMPT_RESOURCES_SQL, (batch_id, 1))
    assert len(rows) == 6
    # a lookup in each of attempt_resources, attempts and resources per row, plus the end of the range
    assert reads <= 3 * len(rows) + 2, reads
