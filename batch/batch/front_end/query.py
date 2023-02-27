import abc
from typing import Any, List, Optional, Tuple, Type

from hailtop.utils import parse_timestamp_msecs

from ..exceptions import QueryError
from .operators import (
    ComparisonOperator,
    GreaterThanEqualOperator,
    GreaterThanOperator,
    LessThanEqualOperator,
    LessThanOperator,
    MatchOperator,
    Operator,
    PartialMatchOperator,
    parse_operator,
)

CURRENT_QUERY_VERSION = 1


state_search_term_to_states = {
    'pending': ['Pending'],
    'ready': ['Ready'],
    'creating': ['Creating'],
    'running': ['Running'],
    'live': ['Ready', 'Creating', 'Running'],
    'cancelled': ['Cancelled'],
    'error': ['Error'],
    'failed': ['Failed'],
    'bad': ['Error', 'Failed'],
    'success': ['Success'],
    'done': ['Cancelled', 'Error', 'Failed', 'Success'],
}


def strip(word: str) -> str:
    return word.lstrip().rstrip()


def parse_int(word: str) -> Optional[int]:
    try:
        return int(word)
    except ValueError:
        return None


def parse_float(word: str) -> Optional[float]:
    try:
        return float(word)
    except ValueError:
        return None


def parse_date(word: str) -> Optional[int]:
    try:
        return parse_timestamp_msecs(word)
    except ValueError:
        return None


def parse_cost(word: str) -> Optional[float]:
    word = word.lstrip('$')
    return parse_float(word)


def parse_search_by_operator_type(
    expected_operator_type: Type[Operator], expected_field: Optional[str], term: str
) -> Optional[Tuple[Operator, str, str]]:
    result = parse_operator(term)
    if result is None:
        return None
    operator, left, right = result
    if expected_field and strip(left) != expected_field:
        return None
    if not isinstance(operator, expected_operator_type):
        return None
    return (operator, strip(left), strip(right))


class Query(abc.ABC):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        raise NotImplementedError

    def query(self) -> Tuple[str, List[Any]]:
        raise NotImplementedError


class StateQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_operator_type(MatchOperator, 'state', term)
        if result is None:
            return False
        _, _, maybe_state = result
        return maybe_state in state_search_term_to_states

    def __init__(self, term: str):
        result = parse_search_by_operator_type(MatchOperator, 'state', term)
        assert result
        op, _, state = result
        self.state = state
        self.op = op

    def query(self) -> Tuple[str, List[Any]]:
        states = state_search_term_to_states[self.state]
        condition = ' OR '.join(['(jobs.state = %s)' for _ in states])
        condition = f'({condition})'
        if self.op.negate:
            condition = f'(NOT {condition})'
        return (condition, states)


class JobIdQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_operator_type(ComparisonOperator, 'job_id', term)
        if result is None:
            return False
        _, _, maybe_job_id = result
        return parse_int(maybe_job_id) is not None

    def __init__(self, term: str):
        result = parse_search_by_operator_type(ComparisonOperator, 'job_id', term)
        assert result
        operator, _, maybe_job_id = result
        job_id = parse_int(maybe_job_id)
        assert job_id is not None
        self.operator = operator
        self.job_id = job_id

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        return (f'(jobs.job_id {op} %s)', [self.job_id])


class InstanceQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_operator_type(MatchOperator, 'instance', term)
        return result is not None

    def __init__(self, term: str):
        result = parse_search_by_operator_type(MatchOperator, 'instance', term)
        assert result
        operator, _, instance = result
        self.instance = instance
        self.operator = operator

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        if isinstance(self.operator, PartialMatchOperator):
            self.instance = f'%{self.instance}%'
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name {op} %s))
'''
        return (sql, [self.instance])


class InstanceCollectionQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_operator_type(MatchOperator, 'instance_collection', term)
        return result is not None

    def __init__(self, term: str):
        result = parse_search_by_operator_type(MatchOperator, 'instance_collection', term)
        assert result
        operator, _, inst_coll = result
        self.inst_coll = inst_coll
        self.operator = operator

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        if isinstance(self.operator, PartialMatchOperator):
            self.inst_coll = f'%{self.inst_coll}%'
        sql = f'(jobs.inst_coll {op} %s)'
        return (sql, [self.inst_coll])


class QuotedExactMatchQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        if len(term) < 3:
            return False
        if term[0] != '"':
            return False
        return term[-1] == '"'

    def __init__(self, term: str):
        assert self.is_valid_query(term)
        self.term = term.lstrip('"').rstrip('"')

    def query(self) -> Tuple[str, List[Any]]:
        sql = '''
(((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s OR `value` = %s)) OR
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name = %s)))
'''
        return (sql, [self.term, self.term, self.term])


class UnquotedPartialMatchQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        if len(term) < 3:
            return False
        if term[0] == '"':
            return False
        return term[-1] != '"'

    def __init__(self, term: str):
        assert self.is_valid_query(term)
        self.term = term

    def query(self) -> Tuple[str, List[Any]]:
        sql = '''
(((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` LIKE %s OR `value` LIKE %s)) OR
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name LIKE %s)))
'''
        escaped_term = f'%{self.term}%'
        return (sql, [escaped_term, escaped_term, escaped_term])


class KeyWordQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        search_result = parse_search_by_operator_type(MatchOperator, None, term)
        if search_result is None:
            return False
        return True

    def __init__(self, term: str):
        result = parse_search_by_operator_type(MatchOperator, None, term)
        assert result
        operator, key, value = result
        self.key = key
        self.value = value
        self.operator = operator

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        if isinstance(self.operator, PartialMatchOperator):
            self.value = f'%{self.value}%'
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s AND `value` {op} %s))
        '''
        return (sql, [self.key, self.value])


class TimeQuery(Query, abc.ABC):
    @classmethod
    def from_data(cls, field: str, operator: Operator, time_msecs: int):
        q = object.__new__(cls)
        q.field = field
        q.operator = operator
        q.time_msecs = time_msecs
        return q

    @staticmethod
    def _is_valid_query(field: str, term: str) -> bool:
        result = parse_search_by_operator_type(ComparisonOperator, field, term)
        if result is None:
            return False
        _, _, maybe_time = result
        return parse_date(maybe_time) is not None

    def __init__(self, field: str, term: str):
        result = parse_search_by_operator_type(ComparisonOperator, field, term)
        assert result
        operator, _, maybe_time = result
        time_msecs = parse_date(maybe_time)
        assert time_msecs is not None
        self.field = field
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE {self.field} {op} %s))
'''
        return (sql, [self.time_msecs])


class StartTimeQuery(TimeQuery):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        return TimeQuery._is_valid_query('start_time', term)

    def __init__(self, term: str):
        super().__init__('start_time', term)


class EndTimeQuery(TimeQuery):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        return TimeQuery._is_valid_query('end_time', term)

    def __init__(self, term: str):
        super().__init__('end_time', term)


class DurationQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_operator_type(ComparisonOperator, 'duration', term)
        if result is None:
            return False
        _, _, maybe_duration = result
        return parse_float(maybe_duration) is not None

    def __init__(self, term):
        result = parse_search_by_operator_type(ComparisonOperator, 'duration', term)
        assert result
        operator, _, time_secs = result
        time_msecs = int(float(time_secs) * 1000 + 1)
        assert time_msecs is not None
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE end_time - start_time {op} %s))
'''
        return (sql, [self.time_msecs])


class CostQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_operator_type(ComparisonOperator, 'cost', term)
        if result is None:
            return False
        _, _, maybe_cost = result
        return parse_float(maybe_cost) is not None

    def __init__(self, term):
        result = parse_search_by_operator_type(ComparisonOperator, 'cost', term)
        assert result
        operator, _, cost_str = result
        cost = parse_float(cost_str)
        assert cost is not None
        self.operator = operator
        self.cost = cost

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        return (f'(cost {op} %s)', [self.cost])


def parse_batch_jobs_query_v1(request, batch_id):
    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s AND batch_updates.committed)']
    where_args = [batch_id]

    last_job_id = request.query.get('last_job_id')
    if last_job_id is not None:
        last_job_id = int(last_job_id)
        where_conditions.append('(jobs.job_id > %s)')
        where_args.append(last_job_id)

    q = request.query.get('q', '')
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
            condition = ' OR '.join(['(jobs.state = %s)' for v in values])
            condition = f'({condition})'
            args = values
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


def parse_batch_jobs_query_v2(request, batch_id):
    queries: List[Query] = []
    q = request.query.get('q', '')

    # logic to make time interval queries fast
    min_start_gt_query: Optional[StartTimeQuery] = None
    max_end_lt_query: Optional[EndTimeQuery] = None

    if q:
        terms = q.lstrip().rstrip().split('\n')
        for term in terms:
            term = term.lstrip().rstrip()
            if InstanceQuery.is_valid_query(term):
                queries.append(InstanceQuery(term))
            elif InstanceCollectionQuery.is_valid_query(term):
                queries.append(InstanceCollectionQuery(term))
            elif JobIdQuery.is_valid_query(term):
                queries.append(JobIdQuery(term))
            elif StateQuery.is_valid_query(term):
                queries.append(StateQuery(term))
            elif StartTimeQuery.is_valid_query(term):
                st_query = StartTimeQuery(term)
                queries.append(st_query)
                if (type(st_query.operator) in [GreaterThanOperator, GreaterThanEqualOperator]) and (
                    min_start_gt_query is None or min_start_gt_query.time_msecs >= st_query.time_msecs
                ):
                    min_start_gt_query = st_query
            elif EndTimeQuery.is_valid_query(term):
                et_query = EndTimeQuery(term)
                queries.append(et_query)
                if (type(et_query.operator) in [LessThanOperator, LessThanEqualOperator]) and (
                    max_end_lt_query is None or max_end_lt_query.time_msecs <= et_query.time_msecs
                ):
                    max_end_lt_query = et_query
            elif DurationQuery.is_valid_query(term):
                queries.append(DurationQuery(term))
            elif CostQuery.is_valid_query(term):
                queries.append(CostQuery(term))
            elif KeyWordQuery.is_valid_query(term):
                queries.append(KeyWordQuery(term))
            elif QuotedExactMatchQuery.is_valid_query(term):
                queries.append(QuotedExactMatchQuery(term))
            elif UnquotedPartialMatchQuery.is_valid_query(term):
                queries.append(UnquotedPartialMatchQuery(term))
            else:
                raise QueryError(f'could not parse term "{term}"')

    # this is to make time interval queries fast by using the bounds on both indices
    if min_start_gt_query and max_end_lt_query and min_start_gt_query.time_msecs <= max_end_lt_query.time_msecs:
        queries.append(StartTimeQuery.from_data('start_time', max_end_lt_query.operator, max_end_lt_query.time_msecs))
        queries.append(EndTimeQuery.from_data('end_time', min_start_gt_query.operator, min_start_gt_query.time_msecs))

    # batch has already been validated
    where_conditions = ['(jobs.batch_id = %s AND batch_updates.committed)']
    where_args = [batch_id]

    cost_conditions = []
    cost_args = []

    last_job_id = request.query.get('last_job_id')
    if last_job_id is not None:
        last_job_id = int(last_job_id)
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


def build_batch_jobs_query(request, batch_id, version):
    if version == 1:
        return parse_batch_jobs_query_v1(request, batch_id)
    return parse_batch_jobs_query_v2(request, batch_id)
