import abc
from typing import Any, List, Optional, Tuple

from hailtop.utils import parse_timestamp_msecs

from ..exceptions import QueryError

CURRENT_QUERY_VERSION = 2

valid_comparison_operators = ['<=', '>=', '!=', '==', '=', '>', '<']

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


def normalize_operator(op: str) -> str:
    assert op in valid_comparison_operators
    if op == '==':
        return '='
    return op


def strip(word: str) -> str:
    return word.lstrip().rstrip()


def is_valid_int(word: str) -> int:
    try:
        int(word)
    except ValueError:
        return False
    return True


def isoformat_to_utc(date_string: str) -> Optional[int]:
    try:
        return parse_timestamp_msecs(date_string)
    except:
        return None


def parse_search_by_field(expected_field: str, term: str) -> Optional[Tuple[str, bool]]:
    if '!=' in term:
        negate = True
        terms = term.split('!=')
    elif '=' in term:
        negate = False
        terms = term.split('=')
    else:
        return None

    if len(terms) != 2:
        return None

    maybe_field, search_word = terms
    if strip(maybe_field) != expected_field:
        return None
    return (strip(search_word), negate)


def parse_search_by_comparison_operator(expected_field: str, term: str) -> Optional[Tuple[str, str]]:
    for maybe_operator in valid_comparison_operators:
        terms = term.split(maybe_operator)
        if len(terms) != 2:
            continue
        maybe_field, search_word = terms
        if strip(maybe_field) != expected_field:
            return None
        maybe_operator = normalize_operator(maybe_operator)
        assert maybe_operator
        return (maybe_operator, strip(search_word))
    return None


def parse_search_by_unknown_field(term: str) -> Optional[Tuple[str, str, bool]]:
    if '!=' in term:
        negate = True
        terms = term.split('!=')
    elif '=' in term:
        negate = False
        terms = term.split('=')
    else:
        return None

    if len(terms) != 2:
        return None
    maybe_field, search_word = terms

    return (strip(maybe_field), strip(search_word), negate)


class Query(abc.ABC):
    tables_queried: List[str]

    @staticmethod
    def is_valid_query(term: str) -> bool:
        raise NotImplementedError

    def query(self) -> Tuple[str, List[Any]]:
        raise NotImplementedError


class StateQuery(Query):
    tables_queried = ['jobs']

    @staticmethod
    def is_valid_query(term: str) -> bool:
        result = parse_search_by_field('state', term)
        if result is None:
            return False
        search_word, _ = result
        return search_word in state_search_term_to_states

    def __init__(self, term):
        result = parse_search_by_field('state', term)
        assert result
        state, negate = result
        self.state = state
        self.negate = negate

    def query(self) -> Tuple[str, List[Any]]:
        states = state_search_term_to_states[self.state]
        condition = ' OR '.join(['(jobs.state = %s)' for _ in states])
        condition = f'({condition})'
        if self.negate:
            condition = f'(NOT {condition})'
        return (condition, states)


class JobIdQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        search_result = parse_search_by_comparison_operator('job_id', term)
        if search_result is None:
            return False
        _, job_id = search_result
        if not is_valid_int(job_id):
            return False
        return True

    def __init__(self, term):
        result = parse_search_by_comparison_operator('job_id', term)
        assert result
        operator, job_id = result
        self.operator = operator
        self.job_id = job_id

    def query(self) -> Tuple[str, List[Any]]:
        return (f'(jobs.job_id {self.operator} %s)', [self.job_id])


class InstanceQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        search_word = parse_search_by_field('instance', term)
        return search_word is not None

    def __init__(self, term):
        result = parse_search_by_field('instance', term)
        assert result
        instance, negate = result
        self.instance = instance
        self.negate = negate

    def query(self) -> Tuple[str, List[Any]]:
        sql = '''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name = %s))
'''
        if self.negate:
            sql = f'(NOT {sql})'
        return (sql, [self.instance])


class InstanceCollectionQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        search_word = parse_search_by_field('instance_collection', term)
        return search_word is not None

    def __init__(self, term):
        result = parse_search_by_field('instance_collection', term)
        assert result
        instance, negate = result
        self.instance = instance
        self.negate = negate

    def query(self) -> Tuple[str, List[Any]]:
        sql = '(jobs.inst_coll = %s)'
        if self.negate:
            sql = f'(NOT {sql})'
        return (sql, [self.instance])


class ExactMatchQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        if len(term) < 3:
            return False
        if term[0] != '"':
            return False
        return term[-1] == '"'

    def __init__(self, term):
        assert self.is_valid_query(term)
        self.term = term

    def query(self) -> Tuple[str, List[Any]]:
        sql = '''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `value` = %s))
'''
        return (sql, [self.term])


class PartialMatchQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        if len(term) < 3:
            return False
        if term[0] == '"':
            return False
        return term[-1] != '"'

    def __init__(self, term):
        assert self.is_valid_query(term)
        self.term = term

    def query(self) -> Tuple[str, List[Any]]:
        sql = '''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `value` LIKE %s))
'''
        return (sql, [f'%{self.term}%'])


class KeyWordQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        search_result = parse_search_by_unknown_field(term)
        if search_result is None:
            return False
        return True

    def __init__(self, term):
        search_result = parse_search_by_unknown_field(term)
        assert search_result
        key, value, negate = search_result
        self.key = key
        self.value = value
        self.negate = negate

    def query(self) -> Tuple[str, List[Any]]:
        operator = '!=' if self.negate else '='
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s AND `value` {operator} %s))
        '''
        return (sql, [self.key, self.value])


class TimeQuery(Query, abc.ABC):
    @staticmethod
    def _is_valid_query(field: str, term: str) -> bool:
        search_result = parse_search_by_comparison_operator(field, term)
        if search_result is None:
            return False
        _, time = search_result
        time_msecs = isoformat_to_utc(time)
        if time_msecs is None:
            return False
        return True

    def __init__(self, field, term):
        self.field = field

        result = parse_search_by_comparison_operator(field, term)
        assert result
        operator, time = result
        time_msecs = isoformat_to_utc(time)
        assert time_msecs is not None

        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[Any]]:
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE {self.field} {self.operator} %s))
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
        search_result = parse_search_by_comparison_operator('duration', term)
        if search_result is None:
            return False
        _, time_secs = search_result
        try:
            float(time_secs)
        except ValueError:
            return False
        return True

    def __init__(self, term):
        result = parse_search_by_comparison_operator('duration', term)
        assert result
        operator, time = result
        time_msecs = int(float(time) * 1000 + 1)
        assert time_msecs is not None
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[Any]]:
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE end_time - start_time {self.operator} %s))
'''
        return (sql, [self.time_msecs])


class CostQuery(Query):
    @staticmethod
    def is_valid_query(term: str) -> bool:
        search_result = parse_search_by_comparison_operator('cost', term)
        if search_result is None:
            return False
        _, cost = search_result
        cost = cost.lstrip('$')
        try:
            float(cost)
        except ValueError:
            return False
        return True

    def __init__(self, term):
        result = parse_search_by_comparison_operator('cost', term)
        assert result
        operator, cost_str = result
        cost = float(cost_str.lstrip('$'))
        self.operator = operator
        self.cost = cost

    def query(self) -> Tuple[str, List[Any]]:
        return (f'(cost {self.operator} %s)', [self.cost])


async def parse_batch_jobs_query_v1(request, batch_id):
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
    for t in terms:
        if t[0] == '!':
            negate = True
            t = t[1:]
        else:
            negate = False

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
                queries.append(StartTimeQuery(term))
            elif EndTimeQuery.is_valid_query(term):
                queries.append(EndTimeQuery(term))
            elif DurationQuery.is_valid_query(term):
                queries.append(DurationQuery(term))
            elif CostQuery.is_valid_query(term):
                queries.append(CostQuery(term))
            elif KeyWordQuery.is_valid_query(term):
                queries.append(KeyWordQuery(term))
            elif ExactMatchQuery.is_valid_query(term):
                queries.append(ExactMatchQuery(term))
            elif PartialMatchQuery.is_valid_query(term):
                queries.append(PartialMatchQuery(term))
            else:
                raise QueryError(f'could not parse term "{term}"')

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
            if isinstance(query, (StartTimeQuery, EndTimeQuery, DurationQuery, InstanceQuery)):
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
