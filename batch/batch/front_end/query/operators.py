import abc
from typing import ClassVar, Set

from ...exceptions import QueryError


class Operator(abc.ABC):
    @abc.abstractmethod
    def to_sql(self) -> str:
        raise NotImplementedError


class ComparisonOperator(Operator, abc.ABC):
    symbols: ClassVar[Set[str]] = {'>=', '>', '<', '<=', '==', '=', '!='}


class MatchOperator(Operator, abc.ABC):
    symbols: ClassVar[Set[str]] = {'=', '!=', '!~', '=~'}


class PartialMatchOperator(MatchOperator, abc.ABC):
    symbols: ClassVar[Set[str]] = {'!~', '=~'}


class ExactMatchOperator(MatchOperator, abc.ABC):
    symbols: ClassVar[Set[str]] = {'=', '!='}


class GreaterThanEqualOperator(ComparisonOperator):
    def to_sql(self) -> str:
        return '>='


class LessThanEqualOperator(ComparisonOperator):
    def to_sql(self) -> str:
        return '<='


class GreaterThanOperator(ComparisonOperator):
    def to_sql(self) -> str:
        return '>'


class LessThanOperator(ComparisonOperator):
    def to_sql(self) -> str:
        return '<'


class NotEqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    def to_sql(self) -> str:
        return '!='


class EqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    def to_sql(self) -> str:
        return '='


class NotEqualPartialMatchOperator(PartialMatchOperator):
    def to_sql(self) -> str:
        return 'NOT LIKE'


class EqualPartialMatchOperator(PartialMatchOperator):
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
