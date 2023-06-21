from typing import Any, List, Optional

from ...exceptions import QueryError
from .query import state_search_term_to_states


def parse_batch_jobs_query_v1(batch_id: int, q: str, last_job_id: Optional[int]):
    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s AND batch_updates.committed)']
    where_args: List[Any] = [batch_id]

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
                condition = '''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s AND `value` = %s))
'''
                args = [k, v]
        elif t.startswith('has:'):
            k = t[4:]
            condition = '''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s))
'''
            args = [k]
        elif t in state_search_term_to_states:
            values = state_search_term_to_states[t]
            condition = ' OR '.join(['(jobs.state = %s)' for _ in values])
            condition = f'({condition})'
            args = [v.value for v in values]
        else:
            raise QueryError(f'Invalid search term: {t}.')

        if negate:
            condition = f'(NOT {condition})'

        where_conditions.append(condition)
        where_args.extend(args)

    sql = f'''
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
SELECT base_t.*, COALESCE(SUM(`usage` * rate), 0) AS cost
FROM base_t
LEFT JOIN (
  SELECT aggregated_job_resources_v2.batch_id, aggregated_job_resources_v2.job_id, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM base_t
  LEFT JOIN aggregated_job_resources_v2 ON base_t.batch_id = aggregated_job_resources_v2.batch_id AND base_t.job_id = aggregated_job_resources_v2.job_id
  GROUP BY aggregated_job_resources_v2.batch_id, aggregated_job_resources_v2.job_id, aggregated_job_resources_v2.resource_id
) AS usage_t ON base_t.batch_id = usage_t.batch_id AND base_t.job_id = usage_t.job_id
LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
GROUP BY base_t.batch_id, base_t.job_id;
'''
    sql_args = where_args

    return (sql, sql_args)
