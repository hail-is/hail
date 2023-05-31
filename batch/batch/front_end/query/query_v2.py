from typing import List, Optional

from ...exceptions import QueryError
from .operators import (
    GreaterThanEqualOperator,
    GreaterThanOperator,
    LessThanEqualOperator,
    LessThanOperator,
    pad_maybe_operator,
)
from .query import (
    CostQuery,
    DurationQuery,
    EndTimeQuery,
    InstanceCollectionQuery,
    InstanceQuery,
    JobIdQuery,
    KeywordQuery,
    Query,
    QuotedExactMatchQuery,
    StartTimeQuery,
    StateQuery,
    UnquotedPartialMatchQuery,
)

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


def parse_batch_jobs_query_v2(batch_id: int, q: str, last_job_id: Optional[int]):
    queries: List[Query] = []

    # logic to make time interval queries fast
    min_start_gt_query: Optional[StartTimeQuery] = None
    max_end_lt_query: Optional[EndTimeQuery] = None

    if q:
        terms = q.split('\n')
        for _term in terms:
            _term = pad_maybe_operator(_term)
            statement = _term.split()
            if len(statement) == 1:
                word = statement[0]
                if word[0] == '"':
                    queries.append(QuotedExactMatchQuery.parse(word))
                else:
                    queries.append(UnquotedPartialMatchQuery.parse(word))
            elif len(statement) == 3:
                left, op, right = statement
                if left == 'instance':
                    queries.append(InstanceQuery.parse(op, right))
                elif left == 'instance_collection':
                    queries.append(InstanceCollectionQuery.parse(op, right))
                elif left == 'job_id':
                    queries.append(JobIdQuery.parse(op, right))
                elif left == 'state':
                    queries.append(StateQuery.parse(op, right))
                elif left == 'start_time':
                    st_query = StartTimeQuery.parse(op, right)
                    queries.append(st_query)
                    if (type(st_query.operator) in [GreaterThanOperator, GreaterThanEqualOperator]) and (
                            min_start_gt_query is None or min_start_gt_query.time_msecs >= st_query.time_msecs
                    ):
                        min_start_gt_query = st_query
                elif left == 'end_time':
                    et_query = EndTimeQuery.parse(op, right)
                    queries.append(et_query)
                    if (type(et_query.operator) in [LessThanOperator, LessThanEqualOperator]) and (
                            max_end_lt_query is None or max_end_lt_query.time_msecs <= et_query.time_msecs
                    ):
                        max_end_lt_query = et_query
                elif left == 'duration':
                    queries.append(DurationQuery.parse(op, right))
                elif left == 'cost':
                    queries.append(CostQuery.parse(op, right))
                else:
                    queries.append(KeywordQuery.parse(op, left, right))
            else:
                raise QueryError(f'could not parse term "{_term}"')

    # this is to make time interval queries fast by using the bounds on both indices
    if min_start_gt_query and max_end_lt_query and min_start_gt_query.time_msecs <= max_end_lt_query.time_msecs:
        queries.append(StartTimeQuery(max_end_lt_query.operator, max_end_lt_query.time_msecs))
        queries.append(EndTimeQuery(min_start_gt_query.operator, min_start_gt_query.time_msecs))

    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s AND batch_updates.committed)']
    where_args = [batch_id]

    cost_conditions = []
    cost_args = []

    if last_job_id is not None:
        where_conditions.append('(jobs.job_id > %s)')
        where_args.append(last_job_id)

    uses_attempts_table = False
    for query in queries:
        cond, args = query.query()
        if isinstance(query, CostQuery):
            cost_conditions.append(f'({cond})')
            cost_args += args
        else:
            if isinstance(
                query,
                (
                    StartTimeQuery,
                    EndTimeQuery,
                    DurationQuery,
                    InstanceQuery,
                    QuotedExactMatchQuery,
                    UnquotedPartialMatchQuery,
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

    if cost_conditions:
        cost_condition_str = f'HAVING {" AND ".join(cost_conditions)}'
    else:
        cost_condition_str = ''

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
  {attempts_table_join_str}
  WHERE {" AND ".join(where_conditions)}
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
GROUP BY base_t.batch_id, base_t.job_id
{cost_condition_str};
'''

    sql_args = where_args + cost_args

    return (sql, sql_args)
