from typing import List, Optional

import hail as hl
from hail.expr import ArrayExpression, BooleanExpression, Expression, StringExpression, StructExpression
from hail.utils.misc import hl_plural, plural


def split_lines(
    row: StructExpression, fields: List[str], *, delimiter: str, missing: str, quote: str
) -> ArrayExpression:
    split_array = row.text._split_line(delimiter, missing=missing, quote=quote, regex=len(delimiter) > 1)
    return (
        hl.case()
        .when(hl.len(split_array) == len(fields), split_array)
        .or_error(
            hl.format(
                f"""error in number of fields found: in file %s
Expected {len(fields)} {plural("field", len(fields))}, found %d %s on line:
%s""",
                row.file,
                hl.len(split_array),
                hl_plural("field", hl.len(split_array)),
                row.text,
            )
        )
    )


def match_comment(comment: str, line: StringExpression) -> Expression:
    if len(comment) == 1:
        return line.startswith(comment)
    return line.matches(comment, True)


def should_remove_line(
    line: StringExpression, *, filter: str, comment: List[str], skip_blank_lines: bool
) -> Optional[BooleanExpression]:
    condition = None
    if filter is not None:
        condition = line.matches(filter)
    if len(comment) > 0:
        if condition is None:
            condition = hl.bool(False)
        for mark in comment:
            condition = condition | match_comment(mark, line)
    if skip_blank_lines:
        if condition is None:
            condition = hl.bool(False)
        condition = condition | (hl.len(line) == 0)
    return condition
