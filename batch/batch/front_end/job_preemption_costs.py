"""Reads a job's billing rows from the batch database and computes its retried cost and projected non-preemptible cost.

The arithmetic lives in batch.preemption_costs; this module only fetches what it needs. Vocabulary follows the Batch
glossary (batch/CONTEXT.md).
"""

from typing import Optional

from gear import Database
from hailtop.batch_client.types import GetJobPreemptionCostsResponseV1Alpha

from ..globals import complete_states
from ..preemption_costs import (
    AttemptResource,
    nonpreemptible_counterparts,
    projected_nonpreemptible_cost,
    retried_attempts_cost,
)

# One row per resource each of the job's attempts was billed for.
ATTEMPT_RESOURCES_SQL = """
SELECT attempts.attempt_id, attempts.reason, attempts.start_time, attempts.rollup_time,
  resources.resource, resources.rate, attempt_resources.quantity
FROM attempt_resources
INNER JOIN attempts
  ON attempts.batch_id = attempt_resources.batch_id AND
     attempts.job_id = attempt_resources.job_id AND
     attempts.attempt_id = attempt_resources.attempt_id
INNER JOIN resources
  ON resources.resource_id = COALESCE(attempt_resources.deduped_resource_id, attempt_resources.resource_id)
WHERE attempt_resources.batch_id = %s AND attempt_resources.job_id = %s;
"""


def _escape_like(s: str) -> str:
    return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


async def get_job_preemption_costs(
    db: Database, batch_id: int, job_id: int, state: str, outcome_attempt_id: Optional[str]
) -> GetJobPreemptionCostsResponseV1Alpha:
    """A job's retried cost and projected non-preemptible cost, each None where it doesn't apply.

    An unfinished job's costs are both None, and nothing is read for it.
    """
    if state not in complete_states:
        return {'retried_attempts_cost': None, 'projected_nonpreemptible_cost': None}

    attempt_resources = [
        AttemptResource(**r)
        async for r in db.select_and_fetchall(
            ATTEMPT_RESOURCES_SQL, (batch_id, job_id), query_name='get_job_attempt_resources'
        )
    ]

    projected_cost = None
    counterparts = sorted(nonpreemptible_counterparts(attempt_resources, outcome_attempt_id))
    if state == 'Success' and counterparts:
        counterpart_resources = [
            (r['resource'], r['rate'])
            async for r in db.select_and_fetchall(
                f"""
SELECT resource, rate
FROM resources
WHERE {' OR '.join(['resource LIKE %s'] * len(counterparts))};
""",
                [f'{_escape_like(product)}/%' for product in counterparts],
                query_name='get_nonpreemptible_counterpart_rates',
            )
        ]
        projected_cost = projected_nonpreemptible_cost(attempt_resources, outcome_attempt_id, counterpart_resources)

    return {
        'retried_attempts_cost': retried_attempts_cost(attempt_resources, outcome_attempt_id),
        'projected_nonpreemptible_cost': projected_cost,
    }
