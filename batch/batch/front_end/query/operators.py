import abc

from ...exceptions import QueryError


class Operator(abc.ABC):
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
    def to_sql(self):
        return '!='


class EqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    def to_sql(self):
        return '='


class NotEqualPartialMatchOperator(PartialMatchOperator):
    def to_sql(self):
        return 'NOT LIKE'


class EqualPartialMatchOperator(PartialMatchOperator):
    def to_sql(self):
        return 'LIKE'


symbols_to_operator = {
    '>=': GreaterThanEqualOperator(),
    '<=': LessThanEqualOperator(),
    '>': GreaterThanOperator(),
    '<': LessThanOperator(),
    '!=': NotEqualExactMatchOperator(),
    '==': EqualExactMatchOperator(),
    '=': EqualExactMatchOperator(),
    '!~': NotEqualPartialMatchOperator(),
    '=~': EqualPartialMatchOperator(),
}


def get_operator(symbol: str) -> Operator:
    if symbol not in symbols_to_operator:
        raise QueryError(f'unknown operator {symbol}')
    return symbols_to_operator[symbol]
