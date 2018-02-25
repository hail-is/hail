from parsimonious import Grammar, NodeVisitor
import hail as hl
from hail.utils.java import unescape_parsable

type_grammar = Grammar(
    r"""
    type = array / set / dict / struct / interval / int64 / int32 / float32 / float64 / bool / str / call / str / locus
    int64 = "int64" / "tint64"
    int32 = "int32" / "tint32" / "int" / "tint"
    float32 = "float32" / "tfloat32"
    float64 = "float64" / "tfloat64" / "tfloat" / "float"
    bool = "tbool" / "bool"
    call = "tcall" / "call"
    str = "tstr" / "str"
    locus = ("tlocus" / "locus") _ "[" _ identifier _ "]"
    array = ("tarray" / "array") _ "<" _ type _ ">"
    set = ("tset" / "set") _ "<" _ type _ ">"
    dict = ("tdict" / "dict") _ "<" type  _ "," _ type ">"
    struct = ("tstruct" / "struct") _ "{" _ fields? _ "}"
    tuple = ("ttuple" / "tuple") _ "(" ((type ("," type)*) / _) ")"
    fields = field _ ("," _ field)*
    field = identifier _ ":" _ type
    interval = ("tinterval" / "interval") _ "<" _ type _ ">"  
    identifier = simple_identifier / escaped_identifier
    simple_identifier = ~"\w+"
    escaped_identifier = "`" ~"([^`\\\\]|\\\\.)*" "`"
    _ = ~"\s*"
    """)


class TypeConstructor(NodeVisitor):
    def visit(self, node):
        return super(TypeConstructor, self).visit(node)

    def generic_visit(self, node, visited_children):
        return visited_children

    def visit_type(self, node, visited_children):
        assert len(visited_children) == 1
        return visited_children[0]

    def visit_int64(self, node, visited_children):
        return hl.tint64

    def visit_int32(self, node, visited_children):
        return hl.tint32

    def visit_float64(self, node, visited_children):
        return hl.tfloat64

    def visit_float32(self, node, visited_children):
        return hl.tfloat32

    def visit_bool(self, node, visited_children):
        return hl.tbool

    def visit_call(self, node, visited_children):
        return hl.tcall

    def visit_str(self, node, visited_children):
        return hl.tstr

    def visit_locus(self, node, visited_children):
        return hl.tlocus(visited_children[4])

    def visit_genomeref(self, node, visited_children):
        return node.text

    def visit_array(self, node, visited_children):
        return hl.tarray(visited_children[4])

    def visit_set(self, node, visited_children):
        return hl.tset(visited_children[4])

    def visit_dict(self, node, visited_children):
        return hl.tdict(visited_children[3], visited_children[7])

    def visit_struct(self, node, visited_children):
        maybe_fds = visited_children[4]
        if not maybe_fds:
            return hl.tstruct([], [])
        else:
            fds = maybe_fds[0]
            names = [f[0] for f in fds]
            types = [f[1] for f in fds]
            return hl.tstruct(names, types)

    def visit_tuple(self, node, visited_children):
        ttuple, _, paren, maybe_types, paren = visited_children
        if not maybe_types:
            return hl.ttuple()
        else:
            [[first, rest]] = maybe_types
            return hl.ttuple(first, *(t for comma, t in rest))


    def visit_fields(self, node, visited_children):
        return [visited_children[0]] + [x[2] for x in visited_children[2]]

    def visit_field(self, node, visited_children):
        return (visited_children[0], visited_children[-1])

    def visit_interval(self, node, visited_children):
        return hl.tinterval(visited_children[4])

    def visit_identifier(self, node, visited_children):
        return visited_children[0]

    def visit_simple_identifier(self, node, visited_children):
        return node.text

    def visit_escaped_identifier(self, node, visited_children):
        return unescape_parsable(node.text[1:-1])


type_node_visitor = TypeConstructor()