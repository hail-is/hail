import abc
from typing import List, Optional, Tuple


class Operator(abc.ABC):
    user_symbols: List[str]
    negate: bool

    @staticmethod
    def is_match(s) -> bool:
        return s in Operator.user_symbols

    @staticmethod
    def to_sql() -> str:
        raise NotImplementedError


class ComparisonOperator(Operator, abc.ABC):
    user_symbols: List[str]
    negate = False

    def to_sql(self):
        assert len(self.user_symbols) == 1
        return self.user_symbols[0]


class MatchOperator(Operator, abc.ABC):
    pass


class PartialMatchOperator(MatchOperator, abc.ABC):
    def to_sql(self):
        if self.negate:
            return 'NOT LIKE'
        return 'LIKE'


class ExactMatchOperator(MatchOperator, abc.ABC):
    def to_sql(self):
        if self.negate:
            return '!='
        return '='


class GreaterThanEqualOperator(ComparisonOperator):
    user_symbols = ['>=']


class LessThanEqualOperator(ComparisonOperator):
    user_symbols = ['<=']


class GreaterThanOperator(ComparisonOperator):
    user_symbols = ['>']


class LessThanOperator(ComparisonOperator):
    user_symbols = ['<']


class NotEqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    user_symbols = ['!=']
    negate = True


class EqualExactMatchOperator(ExactMatchOperator, ComparisonOperator):
    user_symbols = ['==', '=']
    negate = False


class NotEqualPartialMatchOperator(PartialMatchOperator):
    user_symbols = ['!~']
    negate = True


class EqualPartialMatchOperator(PartialMatchOperator):
    user_symbols = ['=~']
    negate = False


ordered_operators = [
    GreaterThanEqualOperator(),
    LessThanEqualOperator(),
    GreaterThanOperator(),
    LessThanOperator(),
    NotEqualPartialMatchOperator(),
    EqualPartialMatchOperator(),
    NotEqualExactMatchOperator(),
    EqualExactMatchOperator(),
]


def parse_operator(term) -> Optional[Tuple[Operator, str, str]]:
    for op in ordered_operators:
        for symbol in op.user_symbols:
            terms = term.split(symbol)
            if len(terms) == 2:
                return (op, terms[0], terms[1])
    return None
