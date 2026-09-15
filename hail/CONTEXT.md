# Hail Query compiler

The IR that Python programs compile to, and the analyses and rewrites that run over it.
This glossary fixes the vocabulary of binding and scope in that IR.

## Language

### Binding structure

**Frame**:
The node that opens a scope. It declares parameters, records how the aggregation and scan
scopes change on entry, and wraps a body. A Frame is a scope, not an allocation: stream
bodies compile inline into their consumer's loop. Every child whose environment differs
from its parent's is a Frame.
_Avoid_: lambda (reserved for a bindable, separately emitted function), region, block args

**Block**:
An ordered list of lets followed by a result. A Block extends the enclosing scope with its
lets; it never opens one. Nested Blocks flatten into one.
_Avoid_: let-chain, body

**Parameter**:
A name a Frame declares and its parent operation supplies a value for on each entry. The
stream element in a stream map, the row and global in a table map, the accumulator in a
fold.
_Avoid_: block argument, block arg, bound name

**Initialiser**:
The value that seeds a loop parameter once, before the first iteration. Later iterations
receive their values from Recur, which supplies arguments, not initialisers.
_Avoid_: argument, zero, seed

**Let**:
One entry in a Block: a name, the expression it is bound to, and the scope (eval, agg, or
scan) in which the name is visible.
_Avoid_: binding (ambiguous with the environment delta)

**Scope**:
One of the three name spaces a name can be visible in: eval, agg, or scan.
_Avoid_: environment, context

**Signature**:
What an operation declares about one of its Frames: given the operation and the Frame's
parameter names, the bindings the Frame introduces, plus a readable label for each
parameter. One signature per operation per Frame-bearing child.
_Avoid_: binder, shape, block args

**Bindings**:
The change a Frame makes to the environment on entry: the names and types it introduces,
which scope each is visible in, and how the agg and scan scopes open, close, or carry over.
_Avoid_: environment delta, scope entry

**Strict**:
Said of an operation's child position when the operation evaluates that child exactly
once, unconditionally, as part of evaluating itself. A non-strict child may run zero, one,
or many times. Strictness is a property of the operation on its child, about scheduling,
not about which names are in scope.
_Avoid_: eager, lazy

### Forms

**Special form**:
A node that shapes control flow or binding rather than computing a value, such as Frame,
Block, If, or Switch. Everything else is an operation.
_Avoid_: control node, builtin

**Operation**:
A node that computes a value from its operands, such as StreamMap or ApplyBinaryPrimOp. In
weak A-normal form every strict operand of an operation is an atom.
_Avoid_: op, primitive, function
