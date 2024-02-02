from typing import Any, List, Optional, Tuple

from ...constants import ROOT_JOB_GROUP_ID
from ...exceptions import QueryError
from .query import job_state_search_term_to_states


def parse_list_batches_query_v1(user: str, q: str, last_batch_id: Optional[int]) -> Tuple[str, List[Any]]:
    where_conditions = [
        '(billing_project_users.`user` = %s AND billing_project_users.billing_project = batches.billing_project)',
        'NOT deleted',
        'job_groups.job_group_id = %s',
    ]
    where_args: List[Any] = [user, ROOT_JOB_GROUP_ID]

    if last_batch_id is not None:
        where_conditions.append('(job_groups.batch_id < %s)')
        where_args.append(last_batch_id)

    terms = q.split()
    for _t in terms:
        if _t[0] == '!':
            negate = True
            t = _t[1:]
        else:
            negate = False
            t = _t

        if '=' in t:
            k, v = t.split('=', 1)
            condition = """
((job_groups.batch_id, job_groups.job_group_id) IN
 (SELECT batch_id, job_group_id FROM job_group_attributes
  WHERE `key` = %s AND `value` = %s))
"""
            args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = """
((job_groups.batch_id, job_groups.job_group_id) IN
 (SELECT batch_id, job_group_id FROM job_group_attributes
  WHERE `key` = %s))
"""
            args = [k]
        elif t.startswith('user:'):
            k = t[5:]
            condition = """
(batches.`user` = %s)
"""
            args = [k]
        elif t.startswith('billing_project:'):
            k = t[16:]
            condition = """
(billing_projects.name_cs = %s)
"""
            args = [k]
        elif t == 'open':
            condition = "(batches.`state` = 'open')"
            args = []
        elif t == 'closed':
            condition = "(batches.`state` != 'open')"
            args = []
        elif t == 'complete':
            condition = "(batches.`state` = 'complete')"
            args = []
        elif t == 'running':
            condition = "(batches.`state` = 'running')"
            args = []
        elif t == 'cancelled':
            condition = '(job_groups_cancelled.id IS NOT NULL)'
            args = []
        elif t == 'failure':
            condition = '(n_failed > 0)'
            args = []
        elif t == 'success':
            # need complete because there might be no jobs
            condition = "(batches.`state` = 'complete' AND n_succeeded = batches.n_jobs)"
            args = []
        else:
            raise QueryError(f'Invalid search term: {t}.')

        if negate:
            condition = f'(NOT {condition})'

        where_conditions.append(condition)
        where_args.extend(args)

    sql = f"""
WITH base_t AS (
  SELECT batches.*, job_groups.batch_id, job_groups.job_group_id,
    job_groups_cancelled.id IS NOT NULL AS cancelled,
    job_groups_n_jobs_in_complete_states.n_completed,
    job_groups_n_jobs_in_complete_states.n_succeeded,
    job_groups_n_jobs_in_complete_states.n_failed,
    job_groups_n_jobs_in_complete_states.n_cancelled
  FROM job_groups
  LEFT JOIN batches ON batches.id = job_groups.batch_id
  LEFT JOIN billing_projects ON batches.billing_project = billing_projects.name
  LEFT JOIN job_groups_n_jobs_in_complete_states
    ON job_groups.batch_id = job_groups_n_jobs_in_complete_states.id AND job_groups.job_group_id = job_groups_n_jobs_in_complete_states.job_group_id
  LEFT JOIN job_groups_cancelled
    ON job_groups.batch_id = job_groups_cancelled.id AND job_groups.job_group_id = job_groups_cancelled.job_group_id
  STRAIGHT_JOIN billing_project_users ON batches.billing_project = billing_project_users.billing_project
  WHERE {' AND '.join(where_conditions)}
  ORDER BY batch_id DESC
  LIMIT 51
)
SELECT base_t.*, cost_t.cost, cost_t.cost_breakdown
FROM base_t
LEFT JOIN LATERAL (
  SELECT COALESCE(SUM(`usage` * rate), 0) AS cost, JSON_OBJECTAGG(resources.resource, COALESCE(`usage` * rate, 0)) AS cost_breakdown
  FROM (
    SELECT resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
    FROM aggregated_job_group_resources_v3
    WHERE base_t.id = aggregated_job_group_resources_v3.batch_id AND base_t.job_group_id = aggregated_job_group_resources_v3.job_group_id
    GROUP BY resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
) AS cost_t ON TRUE
ORDER BY batch_id DESC;
"""

    return (sql, where_args)


def parse_list_job_groups_query_v1(
    batch_id: int, job_group_id: int, last_child_job_group_id: Optional[int]
) -> Tuple[str, List[Any]]:
    where_conds = [
        '(job_groups.batch_id = %s)',
        '(NOT deleted)',
        '(job_group_self_and_ancestors.ancestor_id = %s AND job_group_self_and_ancestors.level = 1)',
    ]
    sql_args = [batch_id, job_group_id]

    if last_child_job_group_id is not None:
        where_conds.append('(job_groups.job_group_id > %s)')
        sql_args.append(last_child_job_group_id)

    sql = f"""
SELECT job_groups.*,
  job_groups_cancelled.id IS NOT NULL AS cancelled,
  job_groups_n_jobs_in_complete_states.n_completed,
  job_groups_n_jobs_in_complete_states.n_succeeded,
  job_groups_n_jobs_in_complete_states.n_failed,
  job_groups_n_jobs_in_complete_states.n_cancelled,
  cost_t.cost, cost_t.cost_breakdown
FROM job_groups
LEFT JOIN batches ON batches.id = job_groups.batch_id
LEFT JOIN job_group_self_and_ancestors
  ON job_group_self_and_ancestors.batch_id = job_groups.batch_id AND
     job_group_self_and_ancestors.job_group_id = job_groups.job_group_id
LEFT JOIN job_groups_n_jobs_in_complete_states
   ON job_groups.batch_id = job_groups_n_jobs_in_complete_states.id AND
      job_groups.job_group_id = job_groups_n_jobs_in_complete_states.job_group_id
LEFT JOIN job_groups_cancelled
   ON job_groups.batch_id = job_groups_cancelled.id AND
      job_groups.job_group_id = job_groups_cancelled.job_group_id
LEFT JOIN LATERAL (
  SELECT COALESCE(SUM(`usage` * rate), 0) AS cost, JSON_OBJECTAGG(resources.resource, COALESCE(`usage` * rate, 0)) AS cost_breakdown
  FROM (
    SELECT resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
    FROM aggregated_job_group_resources_v3
    WHERE job_groups.batch_id = aggregated_job_group_resources_v3.batch_id AND job_groups.job_group_id = aggregated_job_group_resources_v3.job_group_id
    GROUP BY resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
) AS cost_t ON TRUE
WHERE {' AND '.join(where_conds)}
ORDER BY job_group_id ASC
LIMIT 51;
"""

    return (sql, sql_args)


def parse_job_group_jobs_query_v1(
    batch_id: int, job_group_id: int, q: str, last_job_id: Optional[int], recursive: bool
) -> Tuple[str, List[Any]]:
    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s AND batch_updates.committed)']
    where_args: List[Any] = [batch_id]

    if recursive:
        jg_cond = """
((jobs.batch_id, jobs.job_group_id) IN
 (SELECT batch_id, job_group_id FROM job_group_self_and_ancestors
  WHERE ancestor_id = %s))
"""
    else:
        jg_cond = '(jobs.job_group_id = %s)'

    where_conditions.append(jg_cond)
    where_args.append(job_group_id)

    if last_job_id is not None:
        where_conditions.append('(jobs.job_id > %s)')
        where_args.append(last_job_id)

    terms = q.split()
    for _t in terms:
        if _t[0] == '!':
            negate = True
            t = _t[1:]
        else:
            negate = False
            t = _t

        args: List[Any]

        if '=' in t:
            k, v = t.split('=', 1)
            if k == 'job_id':
                condition = '(jobs.job_id = %s)'
                args = [v]
            else:
                condition = """
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s AND `value` = %s))
"""
                args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = """
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s))
"""
            args = [k]
        elif t in job_state_search_term_to_states:
            values = job_state_search_term_to_states[t]
            condition = ' OR '.join(['(jobs.state = %s)' for _ in values])
            condition = f'({condition})'
            args = [v.value for v in values]
        else:
            raise QueryError(f'Invalid search term: {t}.')

        if negate:
            condition = f'(NOT {condition})'

        where_conditions.append(condition)
        where_args.extend(args)

    sql = f"""
WITH base_t AS
(
  SELECT jobs.*, batches.user, batches.billing_project, batches.format_version,
    job_attributes.value AS name
  FROM jobs
  INNER JOIN batches ON jobs.batch_id = batches.id
  INNER JOIN batch_updates ON jobs.batch_id = batch_updates.batch_id AND jobs.update_id = batch_updates.update_id
  LEFT JOIN job_attributes
  ON jobs.batch_id = job_attributes.batch_id AND
    jobs.job_id = job_attributes.job_id AND
    job_attributes.`key` = 'name'
  WHERE {' AND '.join(where_conditions)}
  LIMIT 50
)
SELECT base_t.*, cost_t.cost, cost_t.cost_breakdown
FROM base_t
LEFT JOIN LATERAL (
SELECT COALESCE(SUM(`usage` * rate), 0) AS cost, JSON_OBJECTAGG(resources.resource, COALESCE(`usage` * rate, 0)) AS cost_breakdown
FROM (SELECT resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_job_resources_v3
  WHERE aggregated_job_resources_v3.batch_id = base_t.batch_id AND aggregated_job_resources_v3.job_id = base_t.job_id
  GROUP BY aggregated_job_resources_v3.resource_id
) AS usage_t
LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
) AS cost_t ON TRUE;
"""

    return (sql, where_args)
