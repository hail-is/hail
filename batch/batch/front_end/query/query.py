import abc
from enum import Enum
from typing import Any, List, Tuple

from hailtop.utils import parse_timestamp_msecs

from ...exceptions import QueryError
from .operators import (
    ComparisonOperator,
    ExactMatchOperator,
    MatchOperator,
    NotEqualExactMatchOperator,
    PartialMatchOperator,
    get_operator,
)


def parse_int(word: str) -> int:
    try:
        return int(word)
    except ValueError as e:
        raise QueryError(f'expected int, but found {word}') from e


def parse_float(word: str) -> float:
    try:
        return float(word)
    except ValueError as e:
        raise QueryError(f'expected float, but found {word}') from e


def parse_date(word: str) -> int:
    try:
        return parse_timestamp_msecs(word)
    except ValueError as e:
        raise QueryError(f'expected date, but found {word}') from e


def parse_cost(word: str) -> float:
    word = word.lstrip('$')
    return parse_float(word)


class Query(abc.ABC):
    @abc.abstractmethod
    def query(self) -> Tuple[str, List[Any]]:
        raise NotImplementedError


class JobState(Enum):
    PENDING = 'Pending'
    READY = 'Ready'
    CREATING = 'Creating'
    RUNNING = 'Running'
    CANCELLED = 'Cancelled'
    ERROR = 'Error'
    FAILED = 'Failed'
    SUCCESS = 'Success'


job_state_search_term_to_states = {
    'pending': [JobState.PENDING],
    'ready': [JobState.READY],
    'creating': [JobState.CREATING],
    'running': [JobState.RUNNING],
    'live': [JobState.READY, JobState.CREATING, JobState.RUNNING],
    'cancelled': [JobState.CANCELLED],
    'error': [JobState.ERROR],
    'failed': [JobState.FAILED],
    'bad': [JobState.ERROR, JobState.FAILED],
    'success': [JobState.SUCCESS],
    'done': [JobState.CANCELLED, JobState.ERROR, JobState.FAILED, JobState.SUCCESS],
}


class JobStateQuery(Query):
    @staticmethod
    def parse(op: str, state: str) -> 'JobStateQuery':
        operator = get_operator(op)
        if not isinstance(operator, ExactMatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ExactMatchOperator.symbols}')
        if state not in job_state_search_term_to_states:
            raise QueryError(f'unknown state "{state}"')
        return JobStateQuery(state, operator)

    def __init__(self, state: str, operator: ExactMatchOperator):
        self.state = state
        self.operator = operator

    def query(self) -> Tuple[str, List[str]]:
        states = [s.value for s in job_state_search_term_to_states[self.state]]
        condition = ' OR '.join(['(jobs.state = %s)' for _ in states])
        condition = f'({condition})'
        if isinstance(self.operator, NotEqualExactMatchOperator):
            condition = f'(NOT {condition})'
        return (condition, states)


class JobIdQuery(Query):
    @staticmethod
    def parse(op: str, maybe_job_id: str) -> 'JobIdQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        job_id = parse_int(maybe_job_id)
        return JobIdQuery(job_id, operator)

    def __init__(self, job_id: int, operator: ComparisonOperator):
        self.job_id = job_id
        self.operator = operator

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        return (f'(jobs.job_id {op} %s)', [self.job_id])


class JobInstanceQuery(Query):
    @staticmethod
    def parse(op: str, instance: str) -> 'JobInstanceQuery':
        operator = get_operator(op)
        if not isinstance(operator, MatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {MatchOperator.symbols}')
        return JobInstanceQuery(instance, operator)

    def __init__(self, instance: str, operator: MatchOperator):
        self.instance = instance
        self.operator = operator

    def query(self) -> Tuple[str, List[str]]:
        op = self.operator.to_sql()
        if isinstance(self.operator, PartialMatchOperator):
            self.instance = f'%{self.instance}%'
        sql = f"""
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name {op} %s))
"""
        return (sql, [self.instance])


class JobInstanceCollectionQuery(Query):
    @staticmethod
    def parse(op: str, instance_collection: str) -> 'JobInstanceCollectionQuery':
        operator = get_operator(op)
        if not isinstance(operator, MatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {MatchOperator.symbols}')
        return JobInstanceCollectionQuery(instance_collection, operator)

    def __init__(self, inst_coll: str, operator: MatchOperator):
        self.inst_coll = inst_coll
        self.operator = operator

    def query(self) -> Tuple[str, List[str]]:
        op = self.operator.to_sql()
        if isinstance(self.operator, PartialMatchOperator):
            self.inst_coll = f'%{self.inst_coll}%'
        sql = f'(jobs.inst_coll {op} %s)'
        return (sql, [self.inst_coll])


class JobQuotedExactMatchQuery(Query):
    @staticmethod
    def parse(term: str) -> 'JobQuotedExactMatchQuery':
        if len(term) < 3:
            raise QueryError(f'expected a string of minimum length 3. Found {term}')
        if term[-1] != '"':
            raise QueryError("expected the last character of the string to be '\"'")
        return JobQuotedExactMatchQuery(term[1:-1])

    def __init__(self, term: str):
        self.term = term

    def query(self) -> Tuple[str, List[str]]:
        sql = """
(((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s OR `value` = %s)) OR
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name = %s)))
"""
        return (sql, [self.term, self.term, self.term])


class JobUnquotedPartialMatchQuery(Query):
    @staticmethod
    def parse(term: str) -> 'JobUnquotedPartialMatchQuery':
        if len(term) < 1:
            raise QueryError(f'expected a string of minimum length 1. Found {term}')
        if term[0] == '"':
            raise QueryError("expected the first character of the string to not be '\"'")
        if term[-1] == '"':
            raise QueryError("expected the last character of the string to not be '\"'")
        return JobUnquotedPartialMatchQuery(term)

    def __init__(self, term: str):
        self.term = term

    def query(self) -> Tuple[str, List[str]]:
        sql = """
(((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` LIKE %s OR `value` LIKE %s)) OR
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE instance_name LIKE %s)))
"""
        escaped_term = f'%{self.term}%'
        return (sql, [escaped_term, escaped_term, escaped_term])


class JobKeywordQuery(Query):
    @staticmethod
    def parse(op: str, key: str, value: str) -> 'JobKeywordQuery':
        operator = get_operator(op)
        if not isinstance(operator, MatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {MatchOperator.symbols}')
        return JobKeywordQuery(operator, key, value)

    def __init__(self, operator: MatchOperator, key: str, value: str):
        self.operator = operator
        self.key = key
        self.value = value

    def query(self) -> Tuple[str, List[str]]:
        op = self.operator.to_sql()
        value = self.value
        if isinstance(self.operator, PartialMatchOperator):
            value = f'%{value}%'
        sql = f"""
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM job_attributes
  WHERE `key` = %s AND `value` {op} %s))
        """
        return (sql, [self.key, value])


class JobStartTimeQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'JobStartTimeQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        time_msecs = parse_date(time)
        return JobStartTimeQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        sql = f"""
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE start_time {op} %s))
"""
        return (sql, [self.time_msecs])


class JobEndTimeQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'JobEndTimeQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        time_msecs = parse_date(time)
        return JobEndTimeQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        sql = f"""
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE end_time {op} %s))
"""
        return (sql, [self.time_msecs])


class JobDurationQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'JobDurationQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        time_msecs = int(parse_float(time) * 1000)
        return JobDurationQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        sql = f"""
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE end_time - start_time {op} %s))
"""
        return (sql, [self.time_msecs])


class JobCostQuery(Query):
    @staticmethod
    def parse(op: str, cost_str: str) -> 'JobCostQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        cost = parse_cost(cost_str)
        return JobCostQuery(operator, cost)

    def __init__(self, operator: ComparisonOperator, cost: float):
        self.operator = operator
        self.cost = cost

    def query(self) -> Tuple[str, List[float]]:
        op = self.operator.to_sql()
        return (f'(cost {op} %s)', [self.cost])


class BatchState(Enum):
    OPEN = 'open'
    CLOSED = 'closed'
    COMPLETE = 'complete'
    RUNNING = 'running'
    CANCELLED = 'cancelled'
    FAILURE = 'failure'
    SUCCESS = 'success'


batch_state_search_term_to_state = {
    'open': BatchState.OPEN,
    'closed': BatchState.CLOSED,
    'complete': BatchState.COMPLETE,
    'running': BatchState.RUNNING,
    'cancelled': BatchState.CANCELLED,
    'failure': BatchState.FAILURE,
    'success': BatchState.SUCCESS,
}


class BatchStateQuery(Query):
    @staticmethod
    def parse(op: str, maybe_state: str) -> 'BatchStateQuery':
        operator = get_operator(op)
        if not isinstance(operator, ExactMatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ExactMatchOperator.symbols}')
        if maybe_state not in batch_state_search_term_to_state:
            raise QueryError(f'unknown state "{maybe_state}"')
        state = batch_state_search_term_to_state[maybe_state]
        return BatchStateQuery(state, operator)

    def __init__(self, state: BatchState, operator: ExactMatchOperator):
        self.state = state
        self.operator = operator

    def query(self) -> Tuple[str, List[Any]]:
        args: List[Any]
        if self.state == BatchState.OPEN:
            condition = "(batches.`state` = 'open')"
            args = []
        elif self.state == BatchState.CLOSED:
            condition = "(batches.`state` != 'open')"
            args = []
        elif self.state == BatchState.COMPLETE:
            condition = "(batches.`state` = 'complete')"
            args = []
        elif self.state == BatchState.RUNNING:
            condition = "(batches.`state` = 'running')"
            args = []
        elif self.state == BatchState.CANCELLED:
            condition = '(job_groups_cancelled.id IS NOT NULL)'
            args = []
        elif self.state == BatchState.FAILURE:
            condition = '(n_failed > 0)'
            args = []
        else:
            assert self.state == BatchState.SUCCESS
            # need complete because there might be no jobs
            condition = "(batches.`state` = 'complete' AND n_succeeded = batches.n_jobs)"
            args = []

        if isinstance(self.operator, NotEqualExactMatchOperator):
            condition = f'(NOT {condition})'

        return (condition, args)


class BatchIdQuery(Query):
    @staticmethod
    def parse(op: str, maybe_batch_id: str) -> 'BatchIdQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        batch_id = parse_int(maybe_batch_id)
        return BatchIdQuery(batch_id, operator)

    def __init__(self, batch_id: int, operator: ComparisonOperator):
        self.batch_id = batch_id
        self.operator = operator

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        return (f'(batches.id {op} %s)', [self.batch_id])


class BatchUserQuery(Query):
    @staticmethod
    def parse(op: str, user: str) -> 'BatchUserQuery':
        operator = get_operator(op)
        if not isinstance(operator, ExactMatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ExactMatchOperator.symbols}')
        return BatchUserQuery(user, operator)

    def __init__(self, user: str, operator: ExactMatchOperator):
        self.user = user
        self.operator = operator

    def query(self) -> Tuple[str, List[str]]:
        op = self.operator.to_sql()
        return (f'(batches.user {op} %s)', [self.user])


class BatchBillingProjectQuery(Query):
    @staticmethod
    def parse(op: str, billing_project: str) -> 'BatchBillingProjectQuery':
        operator = get_operator(op)
        if not isinstance(operator, ExactMatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ExactMatchOperator.symbols}')
        return BatchBillingProjectQuery(billing_project, operator)

    def __init__(self, billing_project: str, operator: ExactMatchOperator):
        self.billing_project = billing_project
        self.operator = operator

    def query(self) -> Tuple[str, List[str]]:
        op = self.operator.to_sql()
        return (f'(batches.billing_project {op} %s)', [self.billing_project])


class JobGroupQuotedExactMatchQuery(Query):
    @staticmethod
    def parse(term: str) -> 'JobGroupQuotedExactMatchQuery':
        if len(term) < 3:
            raise QueryError(f'expected a string of minimum length 3. Found {term}')
        if term[-1] != '"':
            raise QueryError("expected the last character of the string to be '\"'")
        return JobGroupQuotedExactMatchQuery(term[1:-1])

    def __init__(self, term: str):
        self.term = term

    def query(self) -> Tuple[str, List[str]]:
        sql = '''
((job_groups.batch_id, job_groups.job_group_id) IN
 (SELECT batch_id, job_group_id FROM job_group_attributes
  WHERE `key` = %s OR `value` = %s))
"""
        return (sql, [self.term, self.term])


class JobGroupUnquotedPartialMatchQuery(Query):
    @staticmethod
    def parse(term: str) -> 'JobGroupUnquotedPartialMatchQuery':
        if len(term) < 1:
            raise QueryError(f'expected a string of minimum length 1. Found {term}')
        if term[0] == '"':
            raise QueryError("expected the first character of the string to not be '\"'")
        if term[-1] == '"':
            raise QueryError("expected the last character of the string to not be '\"'")
        return JobGroupUnquotedPartialMatchQuery(term)

    def __init__(self, term: str):
        self.term = term

    def query(self) -> Tuple[str, List[str]]:
        sql = '''
((job_groups.batch_id, job_groups.job_group_id) IN
 (SELECT batch_id, job_group_id FROM job_group_attributes
  WHERE `key` LIKE %s OR `value` LIKE %s))
"""
        escaped_term = f'%{self.term}%'
        return (sql, [escaped_term, escaped_term])


class JobGroupKeywordQuery(Query):
    @staticmethod
    def parse(op: str, key: str, value: str) -> 'JobGroupKeywordQuery':
        operator = get_operator(op)
        if not isinstance(operator, MatchOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {MatchOperator.symbols}')
        return JobGroupKeywordQuery(operator, key, value)

    def __init__(self, operator: MatchOperator, key: str, value: str):
        self.operator = operator
        self.key = key
        self.value = value

    def query(self) -> Tuple[str, List[str]]:
        op = self.operator.to_sql()
        value = self.value
        if isinstance(self.operator, PartialMatchOperator):
            value = f'%{value}%'
        sql = f'''
((job_groups.batch_id, job_groups.job_group_id) IN
 (SELECT batch_id, job_group_id FROM job_group_attributes
  WHERE `key` = %s AND `value` {op} %s))
        """
        return (sql, [self.key, value])


class JobGroupStartTimeQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'JobGroupStartTimeQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        time_msecs = parse_date(time)
        return JobGroupStartTimeQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        sql = f'(job_groups.time_created {op} %s)'
        return (sql, [self.time_msecs])


class JobGroupEndTimeQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'JobGroupEndTimeQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        time_msecs = parse_date(time)
        return JobGroupEndTimeQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        sql = f'(job_groups.time_completed {op} %s)'
        return (sql, [self.time_msecs])


class JobGroupDurationQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'JobGroupDurationQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        time_msecs = int(parse_float(time) * 1000)
        return JobGroupDurationQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[int]]:
        op = self.operator.to_sql()
        sql = f'((job_groups.time_completed - job_groups.time_created) {op} %s)'
        return (sql, [self.time_msecs])


class JobGroupCostQuery(Query):
    @staticmethod
    def parse(op: str, cost_str: str) -> 'JobGroupCostQuery':
        operator = get_operator(op)
        if not isinstance(operator, ComparisonOperator):
            raise QueryError(f'unexpected operator "{op}" expected one of {ComparisonOperator.symbols}')
        cost = parse_cost(cost_str)
        return JobGroupCostQuery(operator, cost)

    def __init__(self, operator: ComparisonOperator, cost: float):
        self.operator = operator
        self.cost = cost

    def query(self) -> Tuple[str, List[float]]:
        op = self.operator.to_sql()
        return (f'(cost_t.cost {op} %s)', [self.cost])
