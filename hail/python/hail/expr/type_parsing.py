from parsimonious import Grammar, NodeVisitor
from hail.expr.nat import NatVariable
from . import types
from hail.utils.java import unescape_parsable

type_grammar = Grammar(
    r"""
    type = _ (array / ndarray / set / dict / struct / union / tuple / interval / int64 / int32 / float32 / float64 / bool / str / call / str / locus / void / variable) _
    variable = "?" simple_identifier (":" simple_identifier)?
    void = "void" / "tvoid"
    int64 = "int64" / "tint64"
    int32 = "int32" / "tint32" / "int" / "tint"
    float32 = "float32" / "tfloat32"
    float64 = "float64" / "tfloat64" / "tfloat" / "float"
    bool = "tbool" / "bool"
    call = "tcall" / "call"
    str = "tstr" / "str"
    locus = ("tlocus" / "locus") _ "<" identifier ">"
    array = ("tarray" / "array") _ "<" type ">"
    ndarray = ("tndarray" / "ndarray") _ "<" type "," nat ">"
    set = ("tset" / "set") _ "<" type ">"
    dict = ("tdict" / "dict") _ "<" type "," type ">"
    struct = ("tstruct" / "struct") _ "{" (fields / _) "}"
    union = ("tunion" / "union") _ "{" (fields / _) "}"
    tuple = ("ttuple" / "tuple") _ "(" ((type ("," type)*) / _) ")"
    fields = field ("," field)*
    field = identifier ":" type
    interval = ("tinterval" / "interval") _ "<" type ">"
    identifier = _ (simple_identifier / escaped_identifier) _
    simple_identifier = ~r"\w+"
    escaped_identifier = ~"`([^`\\\\]|\\\\.)*`"
    nat = _ (nat_literal / nat_variable) _
    nat_literal = ~"[0-9]+"
    nat_variable = "?nat"
    _ = ~r"\s*"
    """)


class TypeConstructor(NodeVisitor):
    def generic_visit(self, node, visited_children):
        return visited_children

    def visit_type(self, node, visited_children):
        _, [t], _ = visited_children
        return t

    def visit_variable(self, node, visited_children):
        question, name, cond_opt = visited_children
        cond = None
        if cond_opt:
            colon, cond = cond_opt[0]
        return types.tvariable(name, cond)

    def visit_void(self, node, visited_children):
        return types.tvoid

    def visit_int64(self, node, visited_children):
        return types.tint64

    def visit_int32(self, node, visited_children):
        return types.tint32

    def visit_float64(self, node, visited_children):
        return types.tfloat64

    def visit_float32(self, node, visited_children):
        return types.tfloat32

    def visit_bool(self, node, visited_children):
        return types.tbool

    def visit_call(self, node, visited_children):
        return types.tcall

    def visit_str(self, node, visited_children):
        return types.tstr

    def visit_locus(self, node, visited_children):
        tlocus, _, angle_bracket, gr, angle_bracket = visited_children
        return types.tlocus(gr)

    def visit_array(self, node, visited_children):
        tarray, _, angle_bracket, t, angle_bracket = visited_children
        return types.tarray(t)

    def visit_ndarray(self, node, visited_children):
        tndarray, _, angle_bracket, elem_t, comma, ndim, angle_bracket = visited_children
        return types.tndarray(elem_t, ndim)

    def visit_set(self, node, visited_children):
        tset, _, angle_bracket, t, angle_bracket = visited_children
        return types.tset(t)

    def visit_dict(self, node, visited_children):
        tdict, _, angle_bracket, kt, comma, vt, angle_bracket = visited_children
        return types.tdict(kt, vt)

    def visit_struct(self, node, visited_children):
        tstruct, _, brace, maybe_fields, brace = visited_children
        if not maybe_fields:
            return types.tstruct()
        else:
            fields = maybe_fields[0]
            return types.tstruct(**dict(fields))

    def visit_union(self, node, visited_children):
        tunion, _, brace, maybe_fields, brace = visited_children
        if not maybe_fields:
            return types.tunion()
        else:
            fields = maybe_fields[0]
            return types.tunion(**dict(fields))

    def visit_tuple(self, node, visited_children):
        ttuple, _, paren, [maybe_types], paren = visited_children
        if not maybe_types:
            return types.ttuple()
        else:
            [first, rest] = maybe_types
            return types.ttuple(first, *(t for comma, t in rest))

    def visit_fields(self, node, visited_children):
        first, rest = visited_children
        return [first] + [field for comma, field in rest]

    def visit_field(self, node, visited_children):
        name, comma, type = visited_children
        return (name, type)

    def visit_interval(self, node, visited_children):
        tinterval, _, angle_bracket, point_t, angle_bracket = visited_children
        return types.tinterval(point_t)

    def visit_identifier(self, node, visited_children):
        _, [id], _ = visited_children
        return id

    def visit_simple_identifier(self, node, visited_children):
        return node.text

    def visit_escaped_identifier(self, node, visited_children):
        return unescape_parsable(node.text[1:-1])

    def visit_nat(self, node, visited_children):
        _, [nat], _ = visited_children
        return nat

    def visit_nat_literal(self, node, visited_children):
        return int(node.text)

    def visit_nat_variable(self, node, visited_children):
        return NatVariable()


type_node_visitor = TypeConstructor()
