# RFC: Explicit scope structure in the IR

Status: proposed (2026-09-15)
Glossary: `hail/CONTEXT.md`
Decision record: `hail/docs/adr/0001-explicit-scope-structure.md`

## Problem Statement

A compiler developer who writes a pass over the IR must know, for every operation, which
of its children bind names, what those names are, what types they have, and how the
aggregation and scan scopes change on entry. That knowledge lives in one large case
analysis over every operation, and again in tables inside the pretty printer, the
loop-invariant code motion pass, and each pass that hand-matches name-binding operations.
Adding an operation means updating each copy; a missed copy is a silent bug. Extending the
environment while traversing a Block is quadratic in the number of lets, because the env
for each let is rebuilt from scratch. A developer reading the IR cannot see where a scope
opens without consulting the case analysis, and the pretty printer's SSA layout depends on
its own hand-maintained table of block arguments.

## Solution

Make scope explicit in the tree. A Frame node opens a scope: it holds the parameter names
its parent operation binds, a reference to that operation's Signature, and a body. The
Signature computes the Frame's Bindings from the parent's other children, so a traversal
extends the environment by asking the child, never by matching on the parent's kind.
Block stays an ordered list of lets with a result and never opens a scope. Every child
whose environment differs from its parent's becomes a Frame, including children that
bind nothing but change the aggregation or scan scope. Operation companions keep their
current constructor and extractor arity so existing rewrite rules and lowerings compile
unchanged. The per-operation case analysis, the pretty printer's block-argument table, and
the strictness tables are deleted once every operation carries its Signature.

## User Stories

1. As a compiler developer, I want environment extension to read the bindings off the
   child node, so that I never pattern-match on the parent's kind to build a child's env.
2. As a compiler developer adding a new name-binding operation, I want to declare its
   Frame-bearing child once in the IR generator, so that the parser, type checker, pretty
   printer, and every traversal pick it up without further edits.
3. As a compiler developer, I want a single generated method per operation that says which
   children are strict, so that the pretty printer and code motion passes stop keeping
   their own tables.
4. As a pass author, I want the environment for the k-th let in a Block to be built
   incrementally from the environment for the (k-1)-th, so that traversing a Block is
   linear in its number of lets.
5. As a pass author, I want a sequential traversal primitive that yields each child with
   its environment in order, so that I have a linear path available without changing the
   random-access one.
6. As a pass author, I want the random-access `Bindings.get(parent, i)` to keep working, so
   that passes that need a single child's environment are untouched.
7. As a rewrite-rule author, I want `case StreamMap(a, name, body)` to keep matching with
   the same three elements, so that the seventy-odd existing rule sites compile unchanged.
8. As a rewrite-rule author, I want `StreamMap(a, name, body)` to keep constructing with
   the same three arguments, so that fusion and lowering code is unchanged.
9. As a lowering author, I want to re-bind a body under a different operation by wrapping
   it in a Let of the old parameters, so that lowering a table filter to a stream filter is
   the same one-liner it is today.
10. As a lowering author, I want to reach the Frame itself through a field when I need its
    parameters, so that the common path stays simple and the rare path stays possible.
11. As a reviewer, I want the binding rule for each operation to sit beside that
    operation's declaration in the IR generator, so that I review one place per operation.
12. As a reviewer, I want each operation's migration to be verified by its existing
    parse-and-print round-trip test, so that no new test infrastructure is needed to accept
    a batch.
13. As the type checker, I want to verify that a Frame appears only where its parent
    declares one and that its Signature is the parent's, so that an ill-formed tree fails
    early with a named operation.
14. As the type checker, I want to verify that a Frame's parameter count matches what its
    Signature expects, so that variable-arity operations such as `StreamZip` cannot drift.
15. As the parser, I want the Frame's parameter types computed after its siblings are
    typed, so that trees with untyped references parse as they do today.
16. As the pretty printer, I want the SSA layout to derive block arguments from Frames,
    their labels from the Signature, and strictness from the generated method, so that its
    hand-written table is deleted.
17. As a developer reading SSA-style logs, I want parameters printed as `%elt`, `%accum`,
    `%loopvar` when name preservation is off, so that logs stay as readable as today.
18. As a Python user, I want the text wire format between Python and the JVM unchanged,
    so that no Python release is coupled to this refactor.
19. As the author of the loop-invariant code motion pass, I want one node type to test for
    "opens a scope", so that its frame stack maps one to one onto the tree.
20. As the author of the loop-invariant code motion pass, I want strictness to be a
    property of the parent on its child, so that the Frame carries scoping facts only.
21. As a compiler developer, I want two generic Simplify rules that hoist a Block out of a
    strict child and absorb a Block found in a let value or result, so that the ten
    per-operation Block rules are deleted.
22. As a compiler developer, I want those generic rules to never move a let across a
    scope boundary, so that aggregation semantics are preserved by construction.
23. As a compiler developer, I want `ApplyAggOp` initialisation arguments to be
    zero-parameter Frames carrying the agg-drop transition, so that children that change
    scope without binding names are visible in the tree.
24. As a compiler developer, I want `If`, `Switch`, and `Coalesce` branches left as plain
    expressions in this change, so that rules that collapse a conditional need no
    re-wrapping of the surviving branch.
25. As a compiler developer, I want Signature implementations private to each companion,
    so that nothing outside the type checker and environment extension names one.
26. As a compiler developer, I want `TailLoop` to hold initialisers and its body Frame to
    hold the loop variables and loop name, so that formal and actual are not confused.
27. As a maintainer of the type-algebra work, I want Signature to be a single method
    returning typed Bindings, so that a domain-generic version later is a type parameter
    on that method and nothing built here is wasted.
28. As a developer starting an analysis at a detached Frame, I want its parameters to be
    reported free, so that behaviour matches a detached body today and nothing is
    silently bound.
29. As a compiler developer, I want the old case analysis kept as a fallback during
    migration, so that operations can move in batches while the full test suites stay
    green at every step.
30. As a compiler developer, I want the fallback deleted at the end, so that there is one
    source of truth for binding structure.

## Implementation Decisions

Vocabulary follows the glossary: Frame, Block, Parameter, Initialiser, Let, Scope,
Signature, Bindings, Strict, Special form, Operation.

- **Frame** is a new IR node with three fields: the parameter names, a Signature, and a
  body that is a plain expression. Its type is its body's type. It is the only node that
  opens a scope. It extends the value IR for now; moving it to its own kind alongside
  value, table, and matrix IR is deferred to the A-normal-form work, because thirty-odd
  sites dispatch on IR kind and the type-level guarantee is already provided in practice
  by the companions.
- **Block** is unchanged: lets and a result, no parameters, never opens a scope.
- **Signature** is a trait with two methods: given the parent node and the Frame's
  parameter names, return the typed Bindings the Frame introduces, including the
  aggregation and scan transitions; and given the parameter count, return a readable
  label per parameter for the SSA pretty printer, repeating for variable-arity
  operations. One singleton implementation per operation per Frame-bearing child,
  generated by the IR generator and private to the operation's companion. Signatures carry
  no strictness.
- **Parameter types are computed, not stored.** They are derived from siblings, and the
  parser types siblings before it computes a child's environment, so a stored type could
  not be computed at construction time. Every caller of environment extension already
  holds the parent; what changes is that no caller needs the parent's kind.
- **Environment extension** becomes: if child `i` of the parent is a Frame, evaluate its
  Signature against the parent; otherwise the empty Bindings. During migration a fallback
  to the existing case analysis handles operations not yet moved. A new sequential
  traversal primitive yields each child with its environment in order and extends a
  Block's lets incrementally; the random-access form remains and is documented as linear
  in the let index.
- **Companions** keep their arity. The constructor takes the body as an expression and
  wraps it in a Frame with the operation's Signature. The extractor yields the body
  expression, not the Frame. Code that needs the Frame reads the field. Defining the
  extractor in the companion suppresses the synthesised one, so the generator emits both.
- **`TailLoop`** holds initialisers; its body Frame holds the loop variables followed by
  the loop name. The companion re-exposes the current pattern shape.
- **Strictness** is declared per child in the IR generator and emitted as a per-node
  method. Frame-bearing children are non-strict by declaration; `If`, `Switch`, and
  `Coalesce` branches are marked explicitly. This method replaces the tables in the pretty
  printer and the code motion pass.
- **Which children become Frames**: every child whose environment differs from its
  parent's, including zero-parameter children that change the aggregation or scan scope,
  such as `ApplyAggOp` initialisation arguments. Branches of conditionals do not, because
  they bind nothing and change no scope, and wrapping them would force every
  conditional-collapsing rule to re-wrap its result.
- **Generic Simplify rules**: hoist moves the lets of a Block found in a strict child
  above the parent and leaves the result in place, legal because names are unique and
  strict operands have no specified mutual evaluation order; absorb splices a Block found
  in a let value or a Block result into the enclosing Block. Both key on the child being a
  Block, so no let ever crosses a Frame. They replace the per-operation Block rules and
  land before the Frame change.
- **Pretty printer**: SSA layout derives block arguments from Frames, their labels from
  the Signature, and strictness from the generated method. Its hand-written table is
  deleted.
- **Wire format** is unchanged. Parser and pretty printer go through the companions.
- **Sequencing**: (1) sequential traversal primitive; (2) generic hoist and absorb rules
  plus the generated strictness method; (3) Frame, Signature, generator support, fallback;
  (4) value IR operations; (5) table IR; (6) matrix IR; (7) block-matrix IR;
  (8) zero-parameter Frames for scope-changing children; (9) delete the fallback and the
  case analysis. Each step is a separately reviewable PR; the value IR step may split into
  stream operations then aggregation forms if it proves large.

## Testing Decisions

A good test here asserts external behaviour of the compiler on IR values: that a tree
parses and prints to itself, that a rewrite produces the expected tree, that the full
pipeline still produces the same results. Tests do not inspect Signatures, Frames, or
environment internals directly.

- **Primary seam: parse-and-print round trip in the IR suite.** The existing lists of
  value, table, matrix, and block-matrix IR instances each hold every node with the
  environment its free references need. Each instance is printed, re-parsed, and compared.
  Parsing runs type annotation, which drives environment extension, so a wrong Signature
  surfaces as a reference type mismatch naming the operation. An operation's migration is
  accepted when its existing entry passes unchanged. New Frame-bearing shapes, such as
  bodies that are Blocks with lets, are added to these lists where absent.
- **Second seam: the Simplify suite.** Its existing Block-hoisting expectations become
  the tests for the generic hoist rule and must pass unmodified after the per-operation
  rules are deleted. Absorb gains a handful of cases: a Block in a let value, a Block as a
  Block's result, and a Block under a Frame that must not be hoisted across it.
- **Regression: the full JVM and Python suites.** The change is behaviour-preserving, so
  green on both is the acceptance criterion for every step.
- **Prior art**: the value IR parser test and its sibling table, matrix, and block-matrix
  variants in the IR suite; the stream-fusion cases in the Simplify suite; the ForwardLets
  suite for env-sensitive rewrites.
- **Not added**: a differential suite comparing Frame-derived Bindings against the legacy
  case analysis. The round trip already fails on any divergence at a higher level. It can
  be added if a migration batch proves hard to bisect.

## Out of Scope

- A domain-generic Signature parameterised by an algebra of type-like values. Wanted
  later; nothing here precludes it.
- Making branches of `If`, `Switch`, and `Coalesce` into Frames.
- Making Frame its own IR kind rather than a value IR.
- Parsing the IR directly into weak A-normal form and optimising in that form.
- Separating special forms from operations in the IR generator.
- Changing the Python IR, its binding structure, or the wire format.
- Rewriting passes that hand-match name-binding operations for code generation or
  interpretation. They continue to work through the companions and gain nothing from
  Frames.
- A sum type for lets to represent void-typed effects.
- Emitting Frame bodies as separate JVM methods.

## Further Notes

The design went through params-on-Block twice and settled on a separate Frame both times
for present, countable reasons: re-binding a body under another operation is a one-line
Let rather than a splice, thirty Block match sites keep their arity, no constructor needs
absorb logic, and no runtime guard against nested parameterised Blocks is needed. Under
A-normal form the two nodes coincide and may be merged then. The name Lambda is reserved
for a future bindable, separately emitted function; Region collides with the memory
package. See the ADR for the full option analysis.
