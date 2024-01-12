from typing import Any, List, Optional, Tuple

from ...constants import ROOT_JOB_GROUP_ID
from ...exceptions import QueryError
from .operators import (
    GreaterThanEqualOperator,
    GreaterThanOperator,
    LessThanEqualOperator,
    LessThanOperator,
    pad_maybe_operator,
)
from .query import (
    BatchBillingProjectQuery,
    BatchIdQuery,
    BatchStateQuery,
    BatchUserQuery,
    JobCostQuery,
    JobDurationQuery,
    JobEndTimeQuery,
    JobGroupCostQuery,
    JobGroupDurationQuery,
    JobGroupEndTimeQuery,
    JobGroupKeywordQuery,
    JobGroupQuotedExactMatchQuery,
    JobGroupStartTimeQuery,
    JobGroupUnquotedPartialMatchQuery,
    JobIdQuery,
    JobInstanceCollectionQuery,
    JobInstanceQuery,
    JobKeywordQuery,
    JobQuotedExactMatchQuery,
    JobStartTimeQuery,
    JobStateQuery,
    JobUnquotedPartialMatchQuery,
    Query,
)

# <batches-query-list> ::= "" | <batches-query> "\n" <batches-query-list>
# <batches-query> ::= <batch-id-query> | <state-query> | <start-time-query> | <end-time-query> |
#                     <duration-query> | <cost-query> | <quoted-exact-match-query> | <unquoted-partial-match-query> |
#                     <billing-project-query> | <user-query>
# <exact-match-operator> ::= "=" | "==" | "!="
# <partial-match-operator> ::= "!~" | "=~"
# <match-operator> ::= <exact-match-operator> | <partial-match-operator>
# <comparison-operator> ::= ">=" | "<=" | ">" | "<" | <exact-match-operator>
# <batch-id-query> ::= "batch_id" <comparison-operator> <int>
# <state-query> ::= "state" <exact-match-operator> <str>
# <start-time-query> ::= "start_time" <comparison-operator> <datetime_str>
# <end-time-query> ::= "end_time" <comparison-operator> <datetime_str>
# <duration-query> ::= "duration" <comparison-operator> <float>
# <cost-query> ::= "cost" <comparison-operator> <float>
# <billing-project-query> ::= "billing_project" <exact-match-operator> <str>
# <user-query> ::= "user" <exact-match-operator> <str>
# <quoted-exact-match-query> ::= \" <str> \"
# <unquoted-partial-match-query> ::= <str>


def parse_list_batches_query_v2(user: str, q: str, last_batch_id: Optional[int]) -> Tuple[str, List[Any]]:
    queries: List[Query] = []

    # logic to make time interval queries fast
    min_start_gt_query: Optional[JobGroupStartTimeQuery] = None
    max_end_lt_query: Optional[JobGroupEndTimeQuery] = None

    if q:
        terms = q.rstrip().lstrip().split('\n')
        for _term in terms:
            _term = pad_maybe_operator(_term)
            statement = _term.split()
            if len(statement) == 1:
                word = statement[0]
                if word[0] == '"':
                    queries.append(JobGroupQuotedExactMatchQuery.parse(word))
                else:
                    queries.append(JobGroupUnquotedPartialMatchQuery.parse(word))
            elif len(statement) == 3:
                left, op, right = statement
                if left == 'batch_id':
                    queries.append(BatchIdQuery.parse(op, right))
                elif left == 'billing_project':
                    queries.append(BatchBillingProjectQuery.parse(op, right))
                elif left == 'user':
                    queries.append(BatchUserQuery.parse(op, right))
                elif left == 'state':
                    queries.append(BatchStateQuery.parse(op, right))
                elif left == 'start_time':
                    st_query = JobGroupStartTimeQuery.parse(op, right)
                    queries.append(st_query)
                    if (type(st_query.operator) in [GreaterThanOperator, GreaterThanEqualOperator]) and (
                        min_start_gt_query is None or min_start_gt_query.time_msecs >= st_query.time_msecs
                    ):
                        min_start_gt_query = st_query
                elif left == 'end_time':
                    et_query = JobGroupEndTimeQuery.parse(op, right)
                    queries.append(et_query)
                    if (type(et_query.operator) in [LessThanOperator, LessThanEqualOperator]) and (
                        max_end_lt_query is None or max_end_lt_query.time_msecs <= et_query.time_msecs
                    ):
                        max_end_lt_query = et_query
                elif left == 'duration':
                    queries.append(JobGroupDurationQuery.parse(op, right))
                elif left == 'cost':
                    queries.append(JobGroupCostQuery.parse(op, right))
                else:
                    queries.append(JobGroupKeywordQuery.parse(op, left, right))
            else:
                raise QueryError(f'could not parse term "{_term}"')

    # this is to make time interval queries fast by using the bounds on both indices
    if min_start_gt_query and max_end_lt_query and min_start_gt_query.time_msecs <= max_end_lt_query.time_msecs:
        queries.append(JobGroupStartTimeQuery(max_end_lt_query.operator, max_end_lt_query.time_msecs))
        queries.append(JobGroupEndTimeQuery(min_start_gt_query.operator, min_start_gt_query.time_msecs))

    # batch has already been validated
    where_conditions = ['(billing_project_users.`user` = %s)', 'NOT deleted', 'job_groups.job_group_id = %s']
    where_args: List[Any] = [user, ROOT_JOB_GROUP_ID]

    if last_batch_id is not None:
        where_conditions.append('(job_groups.batch_id < %s)')
        where_args.append(last_batch_id)

    for query in queries:
        cond, args = query.query()
        where_conditions.append(f'({cond})')
        where_args += args

    sql = f"""
SELECT batches.*, cost_t.cost, cost_t.cost_breakdown,
    job_groups_cancelled.id IS NOT NULL AS cancelled,
    job_groups_n_jobs_in_complete_states.n_completed,
    job_groups_n_jobs_in_complete_states.n_succeeded,
    job_groups_n_jobs_in_complete_states.n_failed,
    job_groups_n_jobs_in_complete_states.n_cancelled
FROM job_groups
LEFT JOIN batches ON batches.id = job_groups.batch_id
LEFT JOIN billing_projects ON batches.billing_project = billing_projects.name
LEFT JOIN job_groups_n_jobs_in_complete_states ON job_groups.batch_id = job_groups_n_jobs_in_complete_states.id AND job_groups.job_group_id = job_groups_n_jobs_in_complete_states.job_group_id
LEFT JOIN job_groups_cancelled ON job_groups.batch_id = job_groups_cancelled.id AND job_groups.job_group_id = job_groups_cancelled.job_group_id
STRAIGHT_JOIN billing_project_users ON batches.billing_project = billing_project_users.billing_project
LEFT JOIN LATERAL (
  SELECT COALESCE(SUM(`usage` * rate), 0) AS cost, JSON_OBJECTAGG(resources.resource, COALESCE(`usage` * rate, 0)) AS cost_breakdown
  FROM (
    SELECT batch_id, job_group_id, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
    FROM aggregated_job_group_resources_v3
    WHERE job_groups.batch_id = aggregated_job_group_resources_v3.batch_id AND job_groups.job_group_id = aggregated_job_group_resources_v3.job_group_id
    GROUP BY batch_id, job_group_id, resource_id
  ) AS usage_t
  LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
  GROUP BY batch_id, job_group_id
) AS cost_t ON TRUE
WHERE {' AND '.join(where_conditions)}
ORDER BY batches.id DESC
LIMIT 51;
"""

    return (sql, where_args)


# <jobs-query-list> ::= "" | <jobs-query> "\n" <jobs-query-list>
# <jobs-query> ::= <instance-query> | <instance-collection-query> | <job-id-query> | <state-query> |
#                  <start-time-query> | <end-time-query> | <duration-query> | <cost-query> |
#                  <quoted-exact-match-query> | <unquoted-partial-match-query>
# <exact-match-operator> ::= "=" | "==" | "!="
# <partial-match-operator> ::= "!~" | "=~"
# <match-operator> ::= <exact-match-operator> | <partial-match-operator>
# <comparison-operator> ::= ">=" | "<=" | ">" | "<" | <exact-match-operator>
# <instance-query> ::= "instance" <match-operator> <str>
# <instance-collection-query> ::= "instance_collection" <match-operator> <str>
# <job-id-query> ::= "job_id" <comparison-operator> <int>
# <state-query> ::= "state" <exact-match-operator> <str>
# <start-time-query> ::= "start_time" <comparison-operator> <datetime_str>
# <end-time-query> ::= "end_time" <comparison-operator> <datetime_str>
# <duration-query> ::= "duration" <comparison-operator> <float>
# <cost-query> ::= "cost" <comparison-operator> <float>
# <quoted-exact-match-query> ::= \" <str> \"
# <unquoted-partial-match-query> ::= <str>


def parse_batch_jobs_query_v2(batch_id: int, q: str, last_job_id: Optional[int]) -> Tuple[str, List[Any]]:
    queries: List[Query] = []

    # logic to make time interval queries fast
    min_start_gt_query: Optional[JobStartTimeQuery] = None
    max_end_lt_query: Optional[JobEndTimeQuery] = None

    if q:
        terms = q.rstrip().lstrip().split('\n')
        for _term in terms:
            _term = pad_maybe_operator(_term)
            statement = _term.split()
            if len(statement) == 1:
                word = statement[0]
                if word[0] == '"':
                    queries.append(JobQuotedExactMatchQuery.parse(word))
                else:
                    queries.append(JobUnquotedPartialMatchQuery.parse(word))
            elif len(statement) == 3:
                left, op, right = statement
                if left == 'instance':
                    queries.append(JobInstanceQuery.parse(op, right))
                elif left == 'instance_collection':
                    queries.append(JobInstanceCollectionQuery.parse(op, right))
                elif left == 'job_id':
                    queries.append(JobIdQuery.parse(op, right))
                elif left == 'state':
                    queries.append(JobStateQuery.parse(op, right))
                elif left == 'start_time':
                    st_query = JobStartTimeQuery.parse(op, right)
                    queries.append(st_query)
                    if (type(st_query.operator) in [GreaterThanOperator, GreaterThanEqualOperator]) and (
                        min_start_gt_query is None or min_start_gt_query.time_msecs >= st_query.time_msecs
                    ):
                        min_start_gt_query = st_query
                elif left == 'end_time':
                    et_query = JobEndTimeQuery.parse(op, right)
                    queries.append(et_query)
                    if (type(et_query.operator) in [LessThanOperator, LessThanEqualOperator]) and (
                        max_end_lt_query is None or max_end_lt_query.time_msecs <= et_query.time_msecs
                    ):
                        max_end_lt_query = et_query
                elif left == 'duration':
                    queries.append(JobDurationQuery.parse(op, right))
                elif left == 'cost':
                    queries.append(JobCostQuery.parse(op, right))
                else:
                    queries.append(JobKeywordQuery.parse(op, left, right))
            else:
                raise QueryError(f'could not parse term "{_term}"')

    # this is to make time interval queries fast by using the bounds on both indices
    if min_start_gt_query and max_end_lt_query and min_start_gt_query.time_msecs <= max_end_lt_query.time_msecs:
        queries.append(JobStartTimeQuery(max_end_lt_query.operator, max_end_lt_query.time_msecs))
        queries.append(JobEndTimeQuery(min_start_gt_query.operator, min_start_gt_query.time_msecs))

    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s AND batch_updates.committed)']
    where_args = [batch_id]

    if last_job_id is not None:
        where_conditions.append('(jobs.job_id > %s)')
        where_args.append(last_job_id)

    uses_attempts_table = False
    for query in queries:
        cond, args = query.query()
        if isinstance(
            query,
            (
                JobStartTimeQuery,
                JobEndTimeQuery,
                JobDurationQuery,
                JobInstanceQuery,
                JobQuotedExactMatchQuery,
                JobUnquotedPartialMatchQuery,
            ),
        ):
            uses_attempts_table = True

        where_conditions.append(f'({cond})')
        where_args += args

    if uses_attempts_table:
        attempts_table_join_str = (
            'LEFT JOIN attempts ON jobs.batch_id = attempts.batch_id AND jobs.job_id = attempts.job_id'
        )
    else:
        attempts_table_join_str = ''

    sql = f"""
SELECT jobs.*, batches.user, batches.billing_project, batches.format_version, job_attributes.value AS name, cost_t.cost,
  cost_t.cost_breakdown
FROM jobs
INNER JOIN batches ON jobs.batch_id = batches.id
INNER JOIN batch_updates ON jobs.batch_id = batch_updates.batch_id AND jobs.update_id = batch_updates.update_id
LEFT JOIN job_attributes
  ON jobs.batch_id = job_attributes.batch_id AND
    jobs.job_id = job_attributes.job_id AND
    job_attributes.`key` = 'name'
{attempts_table_join_str}
LEFT JOIN LATERAL (
SELECT COALESCE(SUM(`usage` * rate), 0) AS cost, JSON_OBJECTAGG(resources.resource, COALESCE(`usage` * rate, 0)) AS cost_breakdown
FROM (SELECT aggregated_job_resources_v3.batch_id, aggregated_job_resources_v3.job_id, resource_id, CAST(COALESCE(SUM(`usage`), 0) AS SIGNED) AS `usage`
  FROM aggregated_job_resources_v3
  WHERE aggregated_job_resources_v3.batch_id = jobs.batch_id AND aggregated_job_resources_v3.job_id = jobs.job_id
  GROUP BY aggregated_job_resources_v3.batch_id, aggregated_job_resources_v3.job_id, aggregated_job_resources_v3.resource_id
) AS usage_t
LEFT JOIN resources ON usage_t.resource_id = resources.resource_id
GROUP BY usage_t.batch_id, usage_t.job_id
) AS cost_t ON TRUE
WHERE {" AND ".join(where_conditions)}
LIMIT 50;
"""

    return (sql, where_args)
