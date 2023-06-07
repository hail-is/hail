import abc
import re

from ...exceptions import QueryError


class Operator(abc.ABC):
    regex: re.Pattern

    @abc.abstractmethod
    def to_sql(self) -> str:
        raise NotImplementedError


class ComparisonOperator(Operator, abc.ABC):
    symbols = {'>=', '>', '<', '<=', '==', '=', '!='}


class MatchOperator(Operator, abc.ABC):
    symbols = {'=', '!=', '!~', '=~'}


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

    def to_sql(self) -> str:
        return '!='


class EqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    regex = re.compile('([^=]+)(==|=)(.+)')

    def to_sql(self) -> str:
        return '='


class NotEqualPartialMatchOperator(PartialMatchOperator):
    regex = re.compile('([^!]+)(!~)(.+)')

    def to_sql(self) -> str:
        return 'NOT LIKE'


class EqualPartialMatchOperator(PartialMatchOperator):
    regex = re.compile('([^=]+)(=~)(.+)')

    def to_sql(self) -> str:
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


def pad_maybe_operator(s: str) -> str:
    for symbol in symbols_to_operator:
        if symbol in s:
            padded_symbol = f' {symbol} '
            tokens = [token.strip() for token in s.split(symbol)]
            return padded_symbol.join(tokens)
    return s


def get_operator(symbol: str) -> Operator:
    if symbol not in symbols_to_operator:
        raise QueryError(f'unknown operator {symbol}')
    return symbols_to_operator[symbol]
