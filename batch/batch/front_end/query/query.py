import abc
from enum import Enum
from typing import Any, List, Tuple, Type

from hailtop.utils import parse_timestamp_msecs

from ...exceptions import QueryError
from .operators import (
    ComparisonOperator,
    ExactMatchOperator,
    MatchOperator,
    NotEqualExactMatchOperator,
    Operator,
    PartialMatchOperator,
    get_operator,
)


class State(Enum):
    PENDING = 'Pending'
    READY = 'Ready'
    CREATING = 'Creating'
    RUNNING = 'Running'
    CANCELLED = 'Cancelled'
    ERROR = 'Error'
    FAILED = 'Failed'
    SUCCESS = 'Success'


state_search_term_to_states = {
    'pending': [State.PENDING],
    'ready': [State.READY],
    'creating': [State.CREATING],
    'running': [State.RUNNING],
    'live': [State.READY, State.CREATING, State.RUNNING],
    'cancelled': [State.CANCELLED],
    'error': [State.ERROR],
    'failed': [State.FAILED],
    'bad': [State.ERROR, State.FAILED],
    'success': [State.SUCCESS],
    'done': [State.CANCELLED, State.ERROR, State.FAILED, State.SUCCESS],
}


def parse_int(word: str) -> int:
    try:
        return int(word)
    except ValueError:
        raise QueryError(f'expected int, but found {word}')


def parse_float(word: str) -> float:
    try:
        return float(word)
    except ValueError:
        raise QueryError(f'expected float, but found {word}')


def parse_date(word: str) -> int:
    try:
        return parse_timestamp_msecs(word)
    except ValueError:
        raise QueryError(f'expected date, but found {word}')


def parse_cost(word: str) -> float:
    word = word.lstrip('$')
    return parse_float(word)


def parse_operator(expected_operator_type: Type[Operator], symbol: str) -> Operator:
    operator = get_operator(symbol)
    if not isinstance(operator, expected_operator_type):
        raise QueryError(f'unexpected operator "{symbol}" expected {expected_operator_type}')
    return operator


class Query(abc.ABC):
    @abc.abstractmethod
    def query(self) -> Tuple[str, List[Any]]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def parse(*args) -> 'Query':
        raise NotImplementedError


class StateQuery(Query):
    @staticmethod
    def parse(op: str, state: str) -> 'StateQuery':
        operator = parse_operator(ExactMatchOperator, op)
        assert isinstance(operator, ExactMatchOperator)
        if state not in state_search_term_to_states:
            raise QueryError(f'unknown state "{state}"')
        return StateQuery(state, operator)

    def __init__(self, state: str, operator: ExactMatchOperator):
        self.state = state
        self.operator = operator

    def query(self) -> Tuple[str, List[Any]]:
        states = state_search_term_to_states[self.state]
        condition = ' OR '.join(['(jobs.state = %s)' for _ in states])
        condition = f'({condition})'
        if isinstance(self.operator, NotEqualExactMatchOperator):
            condition = f'(NOT {condition})'
        return (condition, states)


class JobIdQuery(Query):
    @staticmethod
    def parse(op: str, maybe_job_id: str) -> 'JobIdQuery':
        operator = parse_operator(ComparisonOperator, op)
        assert isinstance(operator, ComparisonOperator)
        job_id = parse_int(maybe_job_id)
        return JobIdQuery(job_id, operator)

    def __init__(self, job_id: int, operator: ComparisonOperator):
        self.job_id = job_id
        self.operator = operator

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        return (f'(jobs.job_id {op} %s)', [self.job_id])


class InstanceQuery(Query):
    @staticmethod
    def parse(op: str, instance: str) -> 'InstanceQuery':
        operator = parse_operator(MatchOperator, op)
        assert isinstance(operator, MatchOperator)
        return InstanceQuery(instance, operator)

    def __init__(self, instance: str, operator: MatchOperator):
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
    def parse(op: str, instance_collection: str) -> 'InstanceCollectionQuery':
        operator = parse_operator(MatchOperator, op)
        assert isinstance(operator, MatchOperator)
        return InstanceCollectionQuery(instance_collection, operator)

    def __init__(self, inst_coll: str, operator: MatchOperator):
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
    def parse(term: str) -> 'QuotedExactMatchQuery':
        if len(term) < 3:
            raise QueryError(f'expected a string of minimum length 3. Found {term}')
        if term[-1] != '"':
            raise QueryError("expected the last character of the string to be '\"'")
        return QuotedExactMatchQuery(term[1:-1])

    def __init__(self, term: str):
        self.term = term

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
    def parse(term: str) -> 'UnquotedPartialMatchQuery':
        if len(term) < 1:
            raise QueryError(f'expected a string of minimum length 1. Found {term}')
        if term[0] == '"':
            raise QueryError("expected the first character of the string to not be '\"'")
        if term[-1] == '"':
            raise QueryError("expected the last character of the string to not be '\"'")
        return UnquotedPartialMatchQuery(term)

    def __init__(self, term: str):
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


class KeywordQuery(Query):
    @staticmethod
    def parse(op: str, key: str, value: str) -> 'KeywordQuery':
        operator = parse_operator(MatchOperator, op)
        assert isinstance(operator, MatchOperator)
        return KeywordQuery(operator, key, value)

    def __init__(self, operator: MatchOperator, key: str, value: str):
        self.operator = operator
        self.key = key
        self.value = value

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


class StartTimeQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'StartTimeQuery':
        operator = parse_operator(ComparisonOperator, op)
        assert isinstance(operator, ComparisonOperator)
        time_msecs = parse_date(time)
        return StartTimeQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE start_time {op} %s))
'''
        return (sql, [self.time_msecs])


class EndTimeQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'EndTimeQuery':
        operator = parse_operator(ComparisonOperator, op)
        assert isinstance(operator, ComparisonOperator)
        time_msecs = parse_date(time)
        return EndTimeQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
        self.operator = operator
        self.time_msecs = time_msecs

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        sql = f'''
((jobs.batch_id, jobs.job_id) IN
 (SELECT batch_id, job_id FROM attempts
  WHERE end_time {op} %s))
'''
        return (sql, [self.time_msecs])


class DurationQuery(Query):
    @staticmethod
    def parse(op: str, time: str) -> 'DurationQuery':
        operator = parse_operator(ComparisonOperator, op)
        assert isinstance(operator, ComparisonOperator)
        time_msecs = int(parse_float(time) * 1000 + 1)
        return DurationQuery(operator, time_msecs)

    def __init__(self, operator: ComparisonOperator, time_msecs: int):
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
    def parse(op: str, cost_str: str) -> 'CostQuery':
        operator = parse_operator(ComparisonOperator, op)
        assert isinstance(operator, ComparisonOperator)
        cost = parse_float(cost_str)
        return CostQuery(operator, cost)

    def __init__(self, operator: ComparisonOperator, cost: float):
        self.operator = operator
        self.cost = cost

    def query(self) -> Tuple[str, List[Any]]:
        op = self.operator.to_sql()
        return (f'(cost {op} %s)', [self.cost])
