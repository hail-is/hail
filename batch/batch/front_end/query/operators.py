import abc
import re

from ...exceptions import QueryError


class Operator(abc.ABC):
    regex: re.Pattern

    @abc.abstractmethod
    def to_sql(self) -> str:
        raise NotImplementedError


class ComparisonOperator(Operator, abc.ABC):
    pass


class MatchOperator(Operator, abc.ABC):
    pass


class PartialMatchOperator(MatchOperator, abc.ABC):
    pass


class ExactMatchOperator(MatchOperator, abc.ABC):
    pass


class GreaterThanEqualOperator(ComparisonOperator):
    regex = re.compile('([^>]+)(>=)(.+)')

    def to_sql(self) -> str:
        return '>='


class LessThanEqualOperator(ComparisonOperator):
    regex = re.compile('([^<]+)(<=)(.+)')

    def to_sql(self) -> str:
        return '<='


class GreaterThanOperator(ComparisonOperator):
    regex = re.compile('([^>]+)(>)(.+)')

    def to_sql(self) -> str:
        return '>'


class LessThanOperator(ComparisonOperator):
    regex = re.compile('([^<]+)(<)(.+)')

    def to_sql(self) -> str:
        return '<'


class NotEqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    regex = re.compile('([^!]+)(!=)(.+)')

    def to_sql(self):
        return '!='


class EqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    regex = re.compile('([^=]+)(==|=)(.+)')

    def to_sql(self):
        return '='


class NotEqualPartialMatchOperator(PartialMatchOperator):
    regex = re.compile('([^!]+)(!~)(.+)')

    def to_sql(self):
        return 'NOT LIKE'


class EqualPartialMatchOperator(PartialMatchOperator):
    regex = re.compile('([^=]+)(=~)(.+)')

    def to_sql(self):
        return 'LIKE'


symbols_to_operator = {
    '>=': GreaterThanEqualOperator(),
    '<=': LessThanEqualOperator(),
    '>': GreaterThanOperator(),
    '<': LessThanOperator(),
    '!~': NotEqualPartialMatchOperator(),
    '=~': EqualPartialMatchOperator(),
    '!=': NotEqualExactMatchOperator(),
    '==': EqualExactMatchOperator(),
    '=': EqualExactMatchOperator(),
}


def pad_operator(s: str) -> str:
    for op in symbols_to_operator.values():
        match = op.regex.fullmatch(s)
        if match:
            groups = match.groups()
            if len(groups) != 3:
                raise QueryError(f'could not parse operator in query string "{s}"')
            return ' '.join(groups)
    raise QueryError(f'could not parse operator in query string "{s}"')


def pad_maybe_operator(s: str) -> str:
    for symbol, op in symbols_to_operator.items():
        if symbol in s:
            return pad_operator(s)
    return s


def get_operator(symbol: str) -> Operator:
    if symbol not in symbols_to_operator:
        raise QueryError(f'unknown operator {symbol}')
    return symbols_to_operator[symbol]
